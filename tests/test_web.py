import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from drishyam import findings
from drishyam.web import KEY, MAX_UPLOAD_BYTES, app

from .conftest import add_crack, add_rebar_spall, concrete, threaded_mismatches

client = TestClient(app)


def jpeg(img, q=90):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


@pytest.fixture(scope="module")
def photo():
    return jpeg(add_rebar_spall(concrete(1200, 1600, 21), (800, 600)))


def post(data, ctype="image/jpeg", params="", headers=None):
    return client.post(f"/api/jobs{params}", content=data, headers={"Content-Type": ctype, **(headers or {})})


def lines(resp):
    return [json.loads(x) for x in resp.text.splitlines() if x]


def test_health_and_security_headers():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] and r.json()["opencv"].startswith("5.")
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_job_streams_contract_events(photo):
    r = post(photo, params="?pixel_cap=0.5")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    ev = lines(r)
    assert ev[0]["type"] == "job" and ev[0]["privacy"]["boxes"] == []
    assert ev[1]["type"] == "overview" and base64.b64decode(ev[1]["heatmap_png_b64"])[:4] == b"\x89PNG"
    done = ev[-1]
    assert done["type"] == "done" and findings.verify(done["token"], KEY)["job"] == done["work_order"]["job"]


def test_rejects_bad_uploads(photo):
    assert post(b"hello", "text/plain").status_code == 400
    assert post(b"GIF89a" + b"0" * 100, "image/png").status_code == 400
    assert post(photo[:2000]).status_code == 400
    big = b"\xff\xd8\xff" + b"0" * (MAX_UPLOAD_BYTES + 10)
    assert post(big).status_code == 413
    lie = client.post("/api/jobs", content=photo, headers={"Content-Type": "image/jpeg", "Content-Length": "abc"})
    assert lie.status_code in (400, 413)


@pytest.mark.parametrize("q", ["?pixel_cap=0", "?pixel_cap=2", "?usd_cap=1", "?seconds_cap=999",
                               "?chaos=rm_rf", "?pixel_cap=nan", "?chaos=" + "x" * 100])
def test_rejects_bad_params(photo, q):
    r = post(photo, params=q)
    assert r.status_code == 422 and "error" in r.json()


def test_replay_header_validation(photo):
    def hdr(obj):
        return {"X-Drishyam-Replay": base64.urlsafe_b64encode(json.dumps(obj).encode()).decode().rstrip("=")}

    assert post(photo, headers={"X-Drishyam-Replay": "!!!"}).status_code == 422
    assert post(photo, headers=hdr([["shell", 1, 1, 1]])).status_code == 422
    assert post(photo, headers={"X-Drishyam-Replay": "A" * 20000}).status_code == 422
    ok = post(photo, params="?pixel_cap=0.2", headers=hdr([["gaze", 800, 600, 1]]))
    steps = [e for e in lines(ok) if e["type"] == "step"]
    assert ok.status_code == 200 and steps[0]["cause"] == "replay"


def test_chaos_drills_over_http(photo, monkeypatch):
    import functools

    from drishyam import agent, web

    # Still needed: on this photo the stop rule ends the job after three looks (survey, two escalations), so
    # without eps = 0 neither the spike (injected on look 4) nor the malformed planner (step 6) would fire.
    monkeypatch.setattr(web, "JobConfig", functools.partial(agent.JobConfig, eps=0.0))
    ev = lines(post(photo, params="?pixel_cap=1&chaos=p95_spike,planner_malformed"))
    assert any(e.get("cause") == "p95>SLA" for e in ev)
    assert any(e["type"] == "planner" and e["status"] == "rejected" for e in ev)


def test_review_flow_and_tampering(photo):
    done = lines(post(photo, params="?pixel_cap=1"))[-1]
    pending = [f for f in done["work_order"]["findings"] if f["state"] == "pending_review"]
    assert pending, "fixture should yield a critical finding"
    fid = pending[0]["id"]
    body = {"token": done["token"], "finding_id": fid, "decision": "approve", "reviewer": "Test Engineer"}
    r = client.post("/api/review", json=body)
    assert r.status_code == 200
    wo = r.json()["work_order"]
    assert next(f for f in wo["findings"] if f["id"] == fid)["state"] == "approved"
    assert wo["approvals"][0]["reviewer"] == "Test Engineer"
    assert client.post("/api/review", json={**body, "token": r.json()["token"]}).status_code == 409
    forged = findings.sign(done["work_order"], b"x" * 32)
    assert client.post("/api/review", json={**body, "token": forged}).status_code == 403
    assert client.post("/api/review", json={**body, "decision": "downgrade"}).status_code == 400
    assert client.post("/api/review", json={**body, "finding_id": "../etc"}).status_code == 400
    assert client.post("/api/review", json={**body, "reviewer": "a\nb"}).status_code == 400
    assert client.post("/api/review", content=b"x" * 200_000,
                       headers={"Content-Type": "application/json"}).status_code == 413


def test_people_dominant_image_refused(monkeypatch, photo):
    from drishyam import privacy

    def refuse(img):
        raise privacy.PeopleDominant("a person fills 60% of the frame")

    monkeypatch.setattr(privacy, "privacy_pass", refuse)
    r = post(photo)
    assert r.status_code == 422 and "person" in r.json()["error"]


def test_privacy_blur_applies_before_analysis(monkeypatch):
    from drishyam import privacy

    img = concrete(800, 800, 22)
    img[100:300, 100:300] = (np.indices((200, 200)).sum(0) % 2 * 255)[..., None]  # a sharp pattern
    stats = privacy.privacy_pass(img.copy())
    assert stats["boxes"] == []
    blurred = img.copy()
    privacy.blur_boxes(blurred, [[100, 100, 200, 200]])
    assert blurred[100:300, 100:300].std() < img[100:300, 100:300].std() / 5
    assert np.array_equal(blurred[400:], img[400:])


def test_privacy_detectors_give_each_thread_its_own_result():
    """Concurrent uploads share one detectors(); YuNet's input size and NanoDet's input are per-call
    state. The low NanoDet threshold makes both results non-empty and distinct."""
    from drishyam import privacy

    det = privacy.detectors()
    imgs = [concrete(240, 320, 2), add_crack(concrete(320, 240, 3), (10, 60), n=80)]
    assert threaded_mismatches(lambda x: (det.faces(x), det.objects(x, 0.01)), imgs) == []


def test_kill_endpoint_halts_a_running_job(photo, monkeypatch):
    """TestClient buffers whole streams, so the kill lands first; the job must halt after its first look."""
    from drishyam import web

    token = "K" * 24
    monkeypatch.setattr(web.secrets, "token_urlsafe", lambda n: token)
    monkeypatch.setattr(web.secrets, "token_hex", lambda n: "abc123")  # job id j_abc123
    assert client.post("/api/kill", json={"job": "j_abc123", "kill_token": "x" * 24}).status_code == 200
    untouched = lines(post(photo, params="?pixel_cap=1"))
    assert untouched[-1]["cause"] != "kill-switch"  # a wrong token kills nothing
    assert client.post("/api/kill", json={"job": "j_abc123", "kill_token": token}).status_code == 200
    events = lines(post(photo, params="?pixel_cap=1"))
    looks = [e for e in events if e["type"] == "step" and e["decision"]["action"] != "halt"]
    assert events[0]["job"] == "j_abc123" and events[0]["kill_token"] == token
    assert events[-1]["cause"] == "kill-switch" and events[-1]["status"] == "halted" and len(looks) == 1
    assert client.post("/api/kill", json={"job": "j_zzzzzz", "kill_token": "a" * 24}).status_code == 400
    assert client.post("/api/kill", json={"job": "j_abcdef", "kill_token": "short"}).status_code == 400


def test_signing_key_fails_closed_on_a_deployment(monkeypatch):
    from drishyam import web

    monkeypatch.delenv("DRISHYAM_SIGNING_KEY", raising=False)
    monkeypatch.setenv("VERCEL_ENV", "production")
    with pytest.raises(RuntimeError, match="DRISHYAM_SIGNING_KEY"):
        web._signing_key()
    monkeypatch.setenv("DRISHYAM_SIGNING_KEY", "s" * 48)
    assert web._signing_key() == b"s" * 48
