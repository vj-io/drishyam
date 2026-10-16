"""HTTP API for the browser UI (docs/api-contract.md). Deployed on Vercel via `app.py`.

Trust boundary: everything in a request is hostile until checked. Uploads are
size-capped while streaming, sniffed by magic bytes and pixel-counted before
decode; budget parameters are range-validated; replay decisions are
re-validated; review tokens are HMAC-verified. Errors never echo internals.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import secrets
import threading
from pathlib import Path
from typing import Literal

import cv2
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__, findings, imaging, privacy
from .agent import Job, JobConfig

log = logging.getLogger("drishyam")
MAX_UPLOAD_BYTES = 4_400_000  # Vercel caps function request bodies at 4.5 MB
MAX_REPLAY_HEADER = 16_384
MAX_REVIEW_BYTES = 100_000
IMAGE_TYPES = {"image/jpeg", "image/png"}
CHAOS = {"p95_spike", "planner_timeout", "planner_malformed"}
ROOT = Path(__file__).resolve().parent.parent


def _signing_key() -> bytes:
    key = os.environ.get("DRISHYAM_SIGNING_KEY", "")
    if len(key) >= 32:
        return key.encode()
    log.warning("DRISHYAM_SIGNING_KEY unset or short; using an ephemeral key (tokens die with this process)")
    return secrets.token_bytes(32)


KEY = _signing_key()

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


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": __version__, "opencv": cv2.__version__,
            "build": "COOL" if cv2.__file__.startswith("/opt/cool") else "stock (not COOL)",
            "segmenter": "classical-v1", "max_upload_bytes": MAX_UPLOAD_BYTES, "max_pixels": imaging.MAX_PIXELS}


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
    cfg = JobConfig(pixel_cap=pixel_cap, usd_cap=usd_cap, seconds_cap=seconds_cap, chaos=drills)
    try:
        job = Job(img, cfg, kill=stop.is_set, replay=replay)
    except ValueError as exc:
        return error(422, str(exc))

    async def events():
        it = job.run()
        try:
            while (ev := await run_in_threadpool(next, it, None)) is not None:
                if ev["type"] == "job":
                    ev["privacy"] = priv
                elif ev["type"] == "done":
                    ev["token"] = findings.sign(ev["work_order"], KEY)
                yield json.dumps(ev, separators=(",", ":")) + "\n"
                if await request.is_disconnected():
                    stop.set()  # the viewer pressed KILL: halt within one step
        except Exception:
            log.exception("job %s failed", job.id)
            yield json.dumps({"type": "error", "message": "the job failed; it has been logged"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


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
