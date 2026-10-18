import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from drishyam import findings
from drishyam.governor import USD_PER_HOUR
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
    import platform

    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["ok"] and r.json()["opencv"].startswith("5.")
    h = r.json()
    assert h["host"] == "local" and h["arch"] == platform.machine() and "commit" in h
    assert h["segmenter"] == "classical-v1" and h["segmenter_licence"] == "MIT (no learned weights)"
    assert r.headers["cache-control"] == "no-store" and r.headers["x-content-type-options"] == "nosniff"
    assert client.get("/docs").status_code == 404 and client.get("/openapi.json").status_code == 404


def test_health_names_the_dacl10k_licence(monkeypatch):
    import types

    from drishyam import web

    monkeypatch.setattr(web, "default_segmenter", lambda: types.SimpleNamespace(name="onnx:drishyam_dacl10k.onnx"))
    monkeypatch.setenv("DRISHYAM_COMMIT", "abc1234")
    h = client.get("/api/health").json()
    assert h["segmenter_licence"] == "CC BY-NC 4.0, trained on dacl10k" and h["commit"] == "abc1234"


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


def test_chaos_drills_over_http(scene, monkeypatch):
    import functools

    from drishyam import agent, web

    # The drills need a job of 6+ looks: the scene (several defects) with eps = 0, so the stop rule cannot end it
    # before the spike (injected on look 4) and the malformed planner (step 6).
    monkeypatch.setattr(web, "JobConfig", functools.partial(agent.JobConfig, eps=0.0))
    ev = lines(post(jpeg(scene), params="?pixel_cap=1&chaos=p95_spike,planner_malformed"))
    assert any(e.get("governor", {}).get("cause") == "p95>SLA" for e in ev)
    assert any(e["type"] == "planner" and e["status"] == "rejected" for e in ev)
    assert any(e.get("planner_status", {}).get("status") == "rejected" for e in ev)


def test_jobs_consult_the_configured_planner(scene, monkeypatch):
    import functools

    from drishyam import agent, web

    asked = []

    def planner(digest):
        asked.append(digest["step"])
        return '{"strategy": "continue", "params": {}, "rationale": "numbers look fine"}'

    monkeypatch.setattr(web, "PLANNER", planner)  # what planner.from_env() builds when DRISHYAM_PLANNER_URL is set
    monkeypatch.setattr(web, "JobConfig", functools.partial(agent.JobConfig, eps=0.0))
    ev = lines(post(jpeg(scene), params="?pixel_cap=1"))
    assert asked and asked[0] == 5
    assert any(e["type"] == "planner" and e["status"] == "accepted" for e in ev)
    step5 = next(e for e in ev if e["type"] == "step" and e["step"] == 5)
    assert step5["planner_status"] == {"status": "accepted", "reason": "", "strategy": "continue"}
    assert step5["planner"] == "rule"  # "continue" leaves the rule planner's look in place


def test_pace_ms_pauses_between_looks_but_not_the_job_clock(photo):
    import time

    assert post(photo, params="?pace_ms=1501").status_code == 422
    assert post(photo, params="?pace_ms=-1").status_code == 422
    t0 = time.perf_counter()
    ev = lines(post(photo, params="?pixel_cap=0.3&pace_ms=300"))
    wall = time.perf_counter() - t0
    looks = sum(e["type"] == "step" and e["decision"]["action"] != "halt" for e in ev)
    done = ev[-1]
    assert done["type"] == "done" and looks >= 2
    assert wall >= 0.3 * looks  # it did wait after every look
    assert done["summary"]["seconds"] < wall - 0.25 * looks  # and the wait is not in seconds, or in $
    assert done["summary"]["usd"] < (wall - 0.25 * looks) * USD_PER_HOUR / 3600


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


class FakeTable:
    """The two DynamoDB Table calls the AWS kill store makes, recorded."""

    def __init__(self):
        self.items, self.calls = {}, []

    def put_item(self, Item):
        self.calls.append(("put", Item))
        self.items[Item["k"]] = Item
        return {}

    def get_item(self, Key, ConsistentRead=False):
        self.calls.append(("get", Key, ConsistentRead))
        return {"Item": self.items[Key["k"]]} if Key["k"] in self.items else {}


@pytest.mark.parametrize("store", ["in-process", "dynamodb"])
def test_kill_endpoint_halts_a_running_job(photo, monkeypatch, store):
    """TestClient buffers whole streams, so the kill lands first; the job must halt after its first look.
    With DRISHYAM_KILL_TABLE (AWS) the flag lives in DynamoDB instead of this process; same contract."""
    import time

    from drishyam import web

    table = FakeTable()
    if store == "dynamodb":
        monkeypatch.setattr(web, "KILLS", web.DynamoKills("drishyam-kill", table=table))
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
    if store == "dynamodb":
        puts = [c[1] for c in table.calls if c[0] == "put"]
        gets = [c for c in table.calls if c[0] == "get"]
        assert len(puts) == 2 and all(set(p) == {"k", "exp"} and p["exp"] > time.time() + 250 for p in puts)
        assert gets and all(consistent for _, _, consistent in gets)  # one consistent read per step
        assert all(p["k"].startswith("j_abc123-") and token not in p["k"] for p in puts)  # the token is hashed


def test_a_failed_kill_store_read_ends_the_job_as_an_error_not_a_kill(photo, monkeypatch, caplog):
    """Fail closed (the job stops), but a DynamoDB throttle or IAM error is logged as a failure, never as KILL."""
    import logging

    from drishyam import web

    class Throttled(FakeTable):
        def get_item(self, Key, ConsistentRead=False):
            raise RuntimeError("ProvisionedThroughputExceededException")

    monkeypatch.setattr(web, "KILLS", web.DynamoKills("drishyam-kill", table=Throttled()))
    with caplog.at_level(logging.INFO, logger="drishyam"):
        events = lines(post(photo, params="?pixel_cap=1"))
    assert events[-1]["type"] == "error" and not any(e["type"] == "done" for e in events)
    assert "failed" in caplog.text and "kill-switch" not in caplog.text


def test_in_process_kill_flags_expire():
    import asyncio

    from drishyam import web

    async def check():
        kills = web.LocalKills()
        await kills.set("a", 1, {"ttl": 300})
        await kills.set("b", 1, {"ttl": 0})  # already expired: dropped by the next set
        await kills.set("c", 1, {"ttl": 300})
        return await kills.get("a"), await kills.get("b"), await kills.get("x"), set(kills.flags)

    assert asyncio.run(check()) == (True, False, False, {"a", "c"})


def test_dynamodb_kill_store_imports_boto3_only_when_selected(monkeypatch):
    import sys
    import types

    from drishyam import web

    made = []
    fake = types.SimpleNamespace(resource=lambda name: types.SimpleNamespace(
        Table=lambda table: made.append((name, table)) or FakeTable()))
    monkeypatch.setitem(sys.modules, "boto3", fake)
    web.DynamoKills("drishyam-kill")
    assert made == [("dynamodb", "drishyam-kill")]


def test_signing_key_from_secrets_manager(monkeypatch, caplog):
    import sys
    import types

    from drishyam import web

    secret = {"SecretString": "q" * 64}
    asked = []

    def client(name):
        def get_secret_value(SecretId):
            asked.append((name, SecretId))
            return secret

        return types.SimpleNamespace(get_secret_value=get_secret_value)

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=client))
    monkeypatch.delenv("DRISHYAM_SIGNING_KEY", raising=False)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "drishyam")
    arn = "arn:aws:secretsmanager:us-east-1:111111111111:secret:drishyam-signing-key"
    monkeypatch.setenv("DRISHYAM_SIGNING_KEY_SECRET_ARN", arn)
    assert web._signing_key() == b"q" * 64 and asked == [("secretsmanager", arn)]
    secret["SecretString"] = "short"
    with pytest.raises(RuntimeError, match="SECRET_ARN"):
        web._signing_key()
    monkeypatch.delenv("DRISHYAM_SIGNING_KEY_SECRET_ARN")
    with pytest.raises(RuntimeError, match="DRISHYAM_SIGNING_KEY"):  # Lambda without any key refuses to start
        web._signing_key()
    assert "q" * 64 not in caplog.text


def test_one_emf_line_per_job_on_lambda(photo, monkeypatch, capsys):
    from drishyam import web

    monkeypatch.setattr(web, "HOST", "aws-lambda")
    done = lines(post(photo, params="?pixel_cap=0.2"))[-1]
    emf = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"_aws"')]
    assert len(emf) == 1
    (m,) = emf
    (spec,) = m["_aws"]["CloudWatchMetrics"]
    assert spec["Namespace"] == "Drishyam" and spec["Dimensions"] == [["Status"]]
    assert [x["Name"] for x in spec["Metrics"]] == ["JobSeconds", "PixelsFrac", "Steps", "Findings"]
    s = done["summary"]
    assert (m["Status"], m["JobSeconds"], m["PixelsFrac"], m["Steps"], m["Findings"]) == (
        done["status"], s["seconds"], s["pixels_frac"], s["steps"], s["findings"])
    assert isinstance(m["_aws"]["Timestamp"], int)


def test_signing_key_fails_closed_on_a_deployment(monkeypatch):
    from drishyam import web

    monkeypatch.delenv("DRISHYAM_SIGNING_KEY", raising=False)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "drishyam-web")
    with pytest.raises(RuntimeError, match="DRISHYAM_SIGNING_KEY"):
        web._signing_key()
    monkeypatch.setenv("DRISHYAM_SIGNING_KEY", "s" * 48)
    assert web._signing_key() == b"s" * 48


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "docs/agent-loop.svg").exists(), reason="docs/agent-loop.svg not on this branch")
def test_site_agent_loop_is_the_docs_figure():
    # CloudFront serves only public/, so the site keeps its own copy of FIG. 3.0; it must not drift from the docs one.
    assert (ROOT / "public/docs/agent-loop.svg").read_bytes() == (ROOT / "docs/agent-loop.svg").read_bytes()
