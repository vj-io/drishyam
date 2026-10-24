"""Findings, the human-approval state machine, work orders and their signatures.

The rule this module exists to enforce: a critical finding (exposed rebar, or
spalling above the area threshold) can only leave `pending_review` through a
human decision, and the agent can never make a finding less critical.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime

SEVERITY = {"exposed_rebar": 3.0, "spalling": 2.0, "crack": 1.5, "corrosion": 1.0, "efflorescence": 0.5}
SPALL_CRITICAL_AREA_PX = 40_000  # L0 px^2; there is no scale reference, so no mm threshold
IOU_AGREE = 0.5  # PRD rule 3: a finer look whose mask overlaps the coarser one by less than this disagrees
DISCLAIMER = "AI-assisted, not an inspection. Measurements are image pixels, not millimetres."
TOKEN_TTL_S = 7 * 24 * 3600
REVIEWER_MAX = 64


class ReviewError(Exception):
    status = 400


class ForbiddenTransition(ReviewError):
    status = 403


class NotPending(ReviewError):
    status = 409


class SignatureError(ReviewError):
    status = 403


def is_critical(cls: str, area_px: int) -> bool:
    return cls == "exposed_rebar" or (cls == "spalling" and area_px >= SPALL_CRITICAL_AREA_PX)


def critical_rule(cls: str, area_px: int) -> str:
    """The rule that made a finding critical, in words a reviewer can check against the measurement."""
    if cls == "exposed_rebar":
        return "exposed_rebar: always critical"
    return f"{cls} {area_px:,} px² >= {SPALL_CRITICAL_AREA_PX:,} px²"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Finding:
    id: str
    cls: str
    confidence: float
    measure: dict
    box: list[int]
    step: int
    level: int
    region: str
    critical: bool = False
    state: str = "draft"
    crop_jpeg_b64: str | None = field(default=None, repr=False)
    verify_iou: float | None = None  # mask IoU of the latest cross-level check of this defect; None: never checked
    steps: list[int] = field(default_factory=list, init=False)  # every look that saw it, in order
    # The look that made it critical and the area it measured: `measure` is the finest look's, which may be
    # smaller than the (coarser) look that crossed the threshold.
    critical_step: int | None = field(default=None, init=False)
    critical_area_px: int | None = field(default=None, init=False)
    critical_rule: str | None = field(default=None, init=False)
    verified: bool | None = field(default=None, init=False)  # verify_iou >= IOU_AGREE; False: unconfirmed

    def __post_init__(self) -> None:
        self.critical = is_critical(self.cls, self.measure.get("area_px", 0))
        self.state = "pending_review" if self.critical else "draft"
        self.steps = [self.step]
        if self.critical:
            self.critical_step, self.critical_area_px = self.step, self.measure.get("area_px", 0)
            self.critical_rule = critical_rule(self.cls, self.critical_area_px)
        if self.verify_iou is not None:
            self.set_verify(self.verify_iou)

    def set_verify(self, iou: float) -> None:
        """The latest cross-level check of this defect: `verified` is False when the two looks disagreed."""
        self.verify_iou = float(iou)
        self.verified = self.verify_iou >= IOU_AGREE

    def agent_update(self, *, confidence: float, measure: dict, box: list[int], step: int, level: int,
                     crop_jpeg_b64: str | None = None, verify_iou: float | None = None) -> None:
        """Another look at the same defect joins its evidence trail. `step` stays the look that found it; the box
        grows to the union of every look's box (its extent at every scale); confidence is the best seen; measure,
        level and crop come from the finest look (ties: the latest); `verify_iou` and `verified` come from the
        latest cross-level check. Class and state are not the agent's to change, and criticality only rises: any
        look that measures a critical defect makes it critical."""
        if step not in self.steps:
            self.steps.append(step)
        x0, y0 = min(self.box[0], box[0]), min(self.box[1], box[1])
        self.box = [x0, y0, max(self.box[0] + self.box[2], box[0] + box[2]) - x0,
                    max(self.box[1] + self.box[3], box[1] + box[3]) - y0]
        self.confidence = max(self.confidence, confidence)
        if level <= self.level:
            self.measure, self.level = measure, level
            if crop_jpeg_b64:
                self.crop_jpeg_b64 = crop_jpeg_b64
        if verify_iou is not None:
            self.set_verify(verify_iou)
        if not self.critical and is_critical(self.cls, measure.get("area_px", 0)):
            self.critical, self.state = True, "pending_review"
            self.critical_step, self.critical_area_px = step, measure.get("area_px", 0)
            self.critical_rule = critical_rule(self.cls, self.critical_area_px)

    def to_dict(self, crop: bool = True) -> dict:
        d = asdict(self)
        if not crop:
            d.pop("crop_jpeg_b64")
        return d


def decide(finding: dict, decision: str, actor: str) -> None:
    """The only way out of pending_review. `actor` must be 'human'."""
    if decision not in ("approve", "reject"):
        raise ReviewError("decision must be 'approve' or 'reject'")
    if actor != "human":
        raise ForbiddenTransition("only a human may approve or reject a finding")
    if finding.get("state") != "pending_review":
        raise NotPending(f"finding {finding.get('id')} is {finding.get('state')}, not pending_review")
    finding["state"] = "approved" if decision == "approve" else "rejected"


def work_order(job_id: str, findings: list[Finding], gaze_path: list[list[int]], not_assessed=()) -> dict:
    """`not_assessed`: windows the quality gate refused, [{step, box, reason, action}], so they never read as clean."""
    wo = {
        "id": "wo_" + job_id.removeprefix("j_"),
        "job": job_id,
        "created": now_iso(),
        "findings": [f.to_dict(crop=False) for f in findings],
        "gaze_path": gaze_path,
        "not_assessed": list(not_assessed),
        "approvals": [],
        "disclaimer": DISCLAIMER,
    }
    wo["status"] = order_status(wo)
    return wo


def order_status(wo: dict) -> str:
    return "blocked" if any(f["state"] == "pending_review" for f in wo["findings"]) else "ready"


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign(payload: dict, key: bytes, ttl_s: int = TOKEN_TTL_S) -> str:
    body = _b64(json.dumps({"exp": int(time.time()) + ttl_s, "wo": payload}, separators=(",", ":")).encode())
    mac = hmac.new(key, body.encode(), hashlib.sha256).digest()
    return f"{body}.{_b64(mac)}"


def verify(token: str, key: bytes) -> dict:
    try:
        body, mac = token.split(".")
        expected = hmac.new(key, body.encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _unb64(mac)):
            raise SignatureError("signature does not match")
        data = json.loads(_unb64(body))
    except SignatureError:
        raise
    except Exception as exc:
        raise SignatureError("malformed token") from exc
    if data.get("exp", 0) < time.time():
        raise SignatureError("token expired")
    return data["wo"]


def clean_reviewer(name: str) -> str:
    name = (name or "").strip()
    if not 1 <= len(name) <= REVIEWER_MAX or not name.isprintable():
        raise ReviewError(f"reviewer must be 1-{REVIEWER_MAX} printable characters")
    return name


def apply_review(token: str, key: bytes, finding_id: str, decision: str, reviewer: str) -> tuple[str, dict]:
    """Human review of a signed work order; returns the re-signed order. Each approval records `prev_mac`, the
    MAC of the token it was applied to, so two orders forked from one token show up as siblings."""
    reviewer = clean_reviewer(reviewer)
    wo = verify(token, key)
    match = [f for f in wo["findings"] if f["id"] == finding_id]
    if not match:
        raise ReviewError(f"no finding {finding_id!r} in this work order")
    decide(match[0], decision, actor="human")
    wo["approvals"].append({"finding_id": finding_id, "decision": decision, "reviewer": reviewer, "at": now_iso(),
                            "prev_mac": token.split(".")[1]})
    wo["status"] = order_status(wo)
    return sign(wo, key), wo
