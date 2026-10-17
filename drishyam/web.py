"""HTTP API for the browser UI (docs/api-contract.md). Deployed on Vercel via `app.py`, or on AWS Lambda.

Trust boundary: everything in a request is hostile until checked. Uploads are
size-capped while streaming, sniffed by magic bytes and pixel-counted before
decode; budget parameters are range-validated; replay decisions are
re-validated; review tokens are HMAC-verified. Errors never echo internals.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import logging
import os
import platform
import secrets
import threading
import time
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

os.environ.setdefault("VERCEL_TELEMETRY_DISABLED", "1")  # the cache SDK phones home unless told not to
from vercel.cache import AsyncRuntimeCache

from . import __version__, findings, imaging, planner, privacy
from .agent import Job, JobConfig
from .segment import default_segmenter

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("drishyam")
MAX_UPLOAD_BYTES = 4_400_000  # Vercel caps function request bodies at 4.5 MB
MAX_REPLAY_HEADER = 16_384
MAX_REVIEW_BYTES = 100_000
IMAGE_TYPES = {"image/jpeg", "image/png"}
CHAOS = {"p95_spike", "planner_timeout", "planner_malformed"}
ROOT = Path(__file__).resolve().parent.parent
MAX_PACED_S = 25.0  # viewing pauses per job (pace_ms): with the 30 s seconds cap, still inside a 60 s function limit
HOST = ("aws-lambda" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
        else "vercel" if os.environ.get("VERCEL") else "local")


def _signing_key() -> bytes:
    if arn := os.environ.get("DRISHYAM_SIGNING_KEY_SECRET_ARN"):  # AWS: read once per cold start, never logged
        import boto3  # the AWS path only (extra "aws"; the Lambda Python runtime ships it too)

        key = boto3.client("secretsmanager").get_secret_value(SecretId=arn)["SecretString"]
    else:
        key = os.environ.get("DRISHYAM_SIGNING_KEY", "")
    if len(key) >= 32:
        return key.encode()
    # Deployed: per-instance keys would break review across instances.
    if os.environ.get("VERCEL_ENV") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        raise RuntimeError("a signing key (>= 32 chars) must be set on a deployment: "
                           "DRISHYAM_SIGNING_KEY, or DRISHYAM_SIGNING_KEY_SECRET_ARN on AWS")
    log.warning("DRISHYAM_SIGNING_KEY unset or short; using an ephemeral key (tokens die with this process)")
    return secrets.token_bytes(32)


class DynamoKills:
    """Kill flags in a DynamoDB table (partition key `k`, numeric TTL attribute `exp`): the AWS twin of the
    Runtime Cache. Reads are consistent, so a flag written by /api/kill on one instance is seen on the next step."""

    def __init__(self, name: str, table=None) -> None:
        if table is None:
            import boto3  # the AWS path only

            table = boto3.resource("dynamodb").Table(name)
        self.table = table

    async def get(self, key: str) -> bool:
        r = await run_in_threadpool(self.table.get_item, Key={"k": key}, ConsistentRead=True)
        return "Item" in r

    async def set(self, key: str, _value, options: dict) -> None:
        item = {"k": key, "exp": int(time.time()) + options["ttl"]}
        await run_in_threadpool(self.table.put_item, Item=item)


KEY = _signing_key()
PLANNER = planner.from_env()  # None unless DRISHYAM_PLANNER_URL is set: then every job consults it
# KILL must reach a job running in another function instance; the browser's aborted stream
# does not (neither Vercel's proxy nor a Lambda Function URL passes the disconnect through).
# A shared flag does: DynamoDB on AWS, else the regional Runtime Cache, which falls back to an
# in-process dict when run locally.
KILLS = DynamoKills(os.environ["DRISHYAM_KILL_TABLE"]) if os.environ.get("DRISHYAM_KILL_TABLE") else (
    AsyncRuntimeCache(namespace="drishyam-kill"))
KILL_TTL_S = 300


def _kill_key(job_id: str, kill_token: str) -> str:
    return f"{job_id}-{hashlib.sha256(kill_token.encode()).hexdigest()[:32]}"


def _log_done(job_id: str, ev: dict) -> None:
    """One line per finished job: a CloudWatch EMF record on Lambda (metrics without an API call), else plain."""
    s = ev["summary"]
    if HOST != "aws-lambda":
        log.info("job %s %s cause=%s steps=%s px=%.3f s=%.2f", job_id, ev["status"], ev["cause"],
                 s["steps"], s["pixels_frac"], s["seconds"])
        return
    metrics = [{"Name": "JobSeconds", "Unit": "Seconds"}, {"Name": "PixelsFrac", "Unit": "None"},
               {"Name": "Steps", "Unit": "Count"}, {"Name": "Findings", "Unit": "Count"}]
    print(json.dumps({
        "_aws": {"Timestamp": int(time.time() * 1000),
                 "CloudWatchMetrics": [{"Namespace": "Drishyam", "Dimensions": [["Status"]], "Metrics": metrics}]},
        "Status": ev["status"], "JobSeconds": s["seconds"], "PixelsFrac": s["pixels_frac"], "Steps": s["steps"],
        "Findings": s["findings"], "job": job_id, "cause": ev["cause"],
    }, separators=(",", ":")), flush=True)

app = FastAPI(title="Drishyam", docs_url=None, redoc_url=None, openapi_url=None)


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = ", ".join(".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors())
    return error(422, f"invalid parameters: {fields}")


@app.exception_handler(Exception)
async def _crash(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled error", exc_info=exc)
    return error(500, "internal error")


@app.middleware("http")
async def _api_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response


@app.api_route("/api/health", methods=["GET", "HEAD"])  # HEAD: uptime monitors
def health() -> dict:
    name = default_segmenter().name
    # scripts/install_model.sh installs the dacl10k-trained weights under that name; no other model ships here.
    licence = ("MIT (no learned weights)" if name == "classical-v1" else
               "CC BY-NC 4.0, trained on dacl10k" if "dacl10k" in name else "unknown: not a model this repo ships")
    return {"ok": True, "version": __version__, "opencv": cv2.__version__,
            "build": "COOL" if cv2.__file__.startswith("/opt/cool") else "stock (not COOL)",
            "segmenter": name, "segmenter_licence": licence,
            "max_upload_bytes": MAX_UPLOAD_BYTES, "max_pixels": imaging.MAX_PIXELS,
            "host": HOST, "arch": platform.machine(),
            "commit": os.environ.get("VERCEL_GIT_COMMIT_SHA") or os.environ.get("DRISHYAM_COMMIT") or None}


async def _read_capped(request: Request, cap: int) -> bytes | None:
    """The body, or None as soon as it passes `cap` bytes (never buffers more than cap)."""
    declared = request.headers.get("content-length")
    if declared and (not declared.isdigit() or int(declared) > cap):
        return None
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > cap:
            return None
    return bytes(buf)


def _parse_replay(header: str | None) -> list:
    if not header:
        return []
    if len(header) > MAX_REPLAY_HEADER:
        raise ValueError("replay header too large")
    try:
        data = json.loads(base64.urlsafe_b64decode(header + "=" * (-len(header) % 4)))
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise ValueError("replay header is not base64url JSON") from exc
    return data


@app.post("/api/jobs")
async def create_job(
    request: Request,
    pixel_cap: float = Query(0.30, ge=0.02, le=1.0),
    usd_cap: float = Query(0.05, ge=0.0001, le=0.05),
    seconds_cap: float = Query(20.0, ge=1.0, le=30.0),
    chaos: str = Query("", max_length=64),
    pace_ms: int = Query(0, ge=0, le=1500),
):
    kind = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if kind not in IMAGE_TYPES:
        return error(400, "send the image as the raw body with Content-Type image/jpeg or image/png")
    drills = tuple(sorted({c for c in chaos.split(",") if c}))
    if not set(drills) <= CHAOS:
        return error(422, f"chaos must be a comma list of {sorted(CHAOS)}")
    try:
        replay = _parse_replay(request.headers.get("x-drishyam-replay"))
    except ValueError as exc:
        return error(422, str(exc))
    data = await _read_capped(request, MAX_UPLOAD_BYTES)
    if data is None:
        return error(413, f"image is larger than {MAX_UPLOAD_BYTES} bytes; the page re-encodes big photos for you")
    try:
        img = await run_in_threadpool(imaging.decode_image, data)
        priv = await run_in_threadpool(privacy.privacy_pass, img)
    except imaging.ImageError as exc:
        return error(400, str(exc))
    except privacy.PeopleDominant as exc:
        return error(422, str(exc))

    stop = threading.Event()
    paused = [0.0]  # seconds spent in viewing pauses: the job's clock leaves them out of its seconds and $
    cfg = JobConfig(pixel_cap=pixel_cap, usd_cap=usd_cap, seconds_cap=seconds_cap, chaos=drills)
    try:
        job = Job(img, cfg, kill=stop.is_set, replay=replay, planner=PLANNER,
                  clock=lambda: time.perf_counter() - paused[0])
    except ValueError as exc:
        return error(422, str(exc))

    kill_token = secrets.token_urlsafe(18)
    kill_key = _kill_key(job.id, kill_token)

    async def events():
        it = job.run()
        finished = False

        def line(ev: dict) -> str:
            nonlocal finished
            if ev["type"] == "job":
                ev["privacy"] = priv
                ev["kill_token"] = kill_token
            elif ev["type"] == "done":
                ev["token"] = findings.sign(ev["work_order"], KEY)
                finished = True
                _log_done(job.id, ev)
            return json.dumps(ev, separators=(",", ":")) + "\n"

        try:
            while (ev := await run_in_threadpool(next, it, None)) is not None:
                yield line(ev)
                if ev["type"] != "step":
                    continue
                if ev["decision"]["action"] != "halt":
                    for _ in range(job.followups):  # this look's findings: no computing, so they arrive with it
                        yield line(await run_in_threadpool(next, it))
                    if pace_ms and paused[0] < MAX_PACED_S:  # paced for viewing; the wait is not the job's time
                        t0 = time.perf_counter()
                        await asyncio.sleep(min(pace_ms / 1000, MAX_PACED_S - paused[0]))
                        paused[0] += time.perf_counter() - t0
                # The viewer pressed KILL: the governor halts the job before its next look.
                if await request.is_disconnected() or await KILLS.get(kill_key):
                    stop.set()
        except Exception:  # the job, or a kill-store read: fail closed, and log it as the failure it is
            finished = True
            log.exception("job %s failed", job.id)
            yield json.dumps({"type": "error", "message": "the job failed; it has been logged"}) + "\n"
        finally:
            if not finished:  # the server cancelled the stream because the viewer left or pressed KILL
                stop.set()
                log.info("job %s halted cause=kill-switch (stream closed by the client after step %s)",
                         job.id, getattr(job, "step", 0))

    return StreamingResponse(events(), media_type="application/x-ndjson")


class KillIn(BaseModel):
    job: str = Field(pattern=r"^j_[0-9a-f]{6}$")
    kill_token: str = Field(pattern=r"^[A-Za-z0-9_-]{24}$")


@app.post("/api/kill")
async def kill(request: Request):
    raw = await _read_capped(request, 1_000)
    if raw is None:
        return error(413, "kill body too large")
    try:
        body = KillIn.model_validate_json(raw)
    except ValueError:
        return error(400, "body must be JSON with job and kill_token")
    await KILLS.set(_kill_key(body.job, body.kill_token), 1, {"ttl": KILL_TTL_S})
    return {"ok": True, "job": body.job}


class ReviewIn(BaseModel):
    token: str = Field(min_length=10, max_length=60_000)
    finding_id: str = Field(pattern=r"^f\d{1,4}$")
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=1, max_length=64)


@app.post("/api/review")
async def review(request: Request):
    raw = await _read_capped(request, MAX_REVIEW_BYTES)
    if raw is None:
        return error(413, "review body too large")
    try:
        body = ReviewIn.model_validate_json(raw)
    except ValueError:
        return error(400, "body must be JSON with token, finding_id, decision and reviewer")
    try:
        token, wo = findings.apply_review(body.token, KEY, body.finding_id, body.decision, body.reviewer)
    except findings.ReviewError as exc:
        return error(exc.status, str(exc))
    return {"token": token, "work_order": wo}


if os.environ.get("DRISHYAM_SERVE_STATIC") == "1":  # local dev and e2e tests only; Vercel serves public/ from its CDN
    from fastapi.staticfiles import StaticFiles

    _HEADERS = {h["key"]: h["value"] for rule in json.loads((ROOT / "vercel.json").read_text())["headers"]
                for h in rule["headers"]}

    @app.middleware("http")
    async def _static_headers(request: Request, call_next):
        response = await call_next(request)
        if not request.url.path.startswith("/api/"):
            response.headers.update(_HEADERS)  # same headers as production, so CSP breakage shows up locally
        return response

    app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="static")
