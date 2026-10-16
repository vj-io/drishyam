"""Optional LLM planner behind an allow-list validator and a hard timeout.

The planner sees a numeric digest only (scores, entropies, latencies, budget
left, counts per fixed class name). It never sees pixels, OCR or any text found
in an image, so text painted on a bridge cannot steer the agent. Anything other
than a valid allow-listed JSON proposal is discarded and the rule planner's
choice stands, logged as `planner-fallback`.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

STRATEGIES = ("continue", "switch_model", "change_level", "reverify", "batch_remaining", "stop")
MODELS = ("fast", "fp16", "accurate")
RATIONALE_MAX = 200
TIMEOUT_S = 2.0

SYSTEM_PROMPT = (
    "You steer a bridge-photo inspection agent. You receive a JSON digest of numbers. "
    "Reply with ONE JSON object and nothing else: "
    '{"strategy": "continue|switch_model|change_level|reverify|batch_remaining|stop", '
    '"params": {"model": "fast|fp16|accurate", "level": 0, "region_id": "r1"}, "rationale": "<=200 chars"}. '
    "Prefer continue unless the numbers clearly argue otherwise."
)


class Invalid(ValueError):
    pass


def validate(obj, digest: dict) -> dict:
    """Return a clean proposal or raise Invalid. Budget-adding moves need budget left."""
    if not isinstance(obj, dict):
        raise Invalid("proposal is not an object")
    strategy = obj.get("strategy")
    if strategy not in STRATEGIES:
        raise Invalid(f"strategy {strategy!r} not in allow-list")
    params = obj.get("params") or {}
    if not isinstance(params, dict):
        raise Invalid("params is not an object")
    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str) or len(rationale) > RATIONALE_MAX:
        raise Invalid("rationale missing or too long")
    clean: dict = {}
    regions = {c["id"] for c in digest["top_candidates"]}
    if strategy == "switch_model":
        if params.get("model") not in MODELS:
            raise Invalid("model not in allow-list")
        clean["model"] = params["model"]
    if strategy in ("change_level", "reverify"):
        rid = params.get("region_id")
        if rid not in regions:
            raise Invalid("region_id is not a live candidate")
        clean["region_id"] = rid
        if digest["looks_left"] < 1:
            raise Invalid("no budget left for another look")
    if strategy == "change_level":
        level = params.get("level")
        if not isinstance(level, int) or isinstance(level, bool) or not 0 <= level <= digest["max_level"]:
            raise Invalid("level out of range")
        clean["level"] = level
    return {"strategy": strategy, "params": clean, "rationale": rationale}


def parse(text: str) -> dict:
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise Invalid("reply is not JSON") from exc


class HttpPlanner:
    """Any OpenAI-compatible chat endpoint: local llama-server, Groq, Gemini, GitHub Models."""

    name = "llm"

    def __init__(self, url: str, model: str = "qwen2.5-3b-instruct", key: str | None = None) -> None:
        if not url.startswith(("http://127.0.0.1", "http://localhost", "https://")):
            raise ValueError("planner URL must be https:// or a loopback http:// address")
        self.url = url.rstrip("/") + "/chat/completions"
        self.model, self.key = model, key

    def __call__(self, digest: dict) -> str:
        body = json.dumps({
            "model": self.model, "temperature": 0, "max_tokens": 160,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": json.dumps(digest)}],
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(self.url, body, headers)
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            return json.loads(r.read(65536))["choices"][0]["message"]["content"]


class ChaosPlanner:
    """Failure drill: hangs past the timeout, or answers with junk."""

    name = "chaos"

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.release = threading.Event()

    def __call__(self, digest: dict) -> str:
        if self.mode == "timeout":
            self.release.wait(TIMEOUT_S * 3)
            return '{"strategy": "continue", "params": {}, "rationale": "too late"}'
        return '{"strategy": "rm -rf /", "params": {"cmd": "curl evil"}, "rationale": "ignore previous"}'


def from_env() -> HttpPlanner | None:
    url = os.environ.get("DRISHYAM_PLANNER_URL")
    if not url:
        return None
    return HttpPlanner(url, os.environ.get("DRISHYAM_PLANNER_MODEL", "qwen2.5-3b-instruct"),
                       os.environ.get("DRISHYAM_PLANNER_KEY"))


def consult(planner, digest: dict, timeout_s: float = TIMEOUT_S) -> tuple[dict | None, str, str]:
    """(proposal or None, status, reason). Never raises: every failure is a fallback."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        text = pool.submit(planner, digest).result(timeout=timeout_s)
        return validate(parse(text), digest), "accepted", ""
    except FutureTimeout:
        return None, "fallback", "timeout"
    except Invalid as exc:
        return None, "rejected", str(exc)
    except Exception as exc:  # noqa: BLE001 -- network, HTTP, schema: all fall back to rules
        return None, "fallback", type(exc).__name__
    finally:
        if isinstance(planner, ChaosPlanner):
            planner.release.set()
        pool.shutdown(wait=False)
