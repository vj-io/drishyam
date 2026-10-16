import time

import pytest

from drishyam import findings as F

KEY = b"k" * 32


def make(cls="exposed_rebar", area=500, fid="f1"):
    return F.Finding(fid, cls, 0.8, {"length_px": 10.0, "width_px": 2.0, "area_px": area, "orientation_deg": 0.0},
                     [0, 0, 10, 10], 3, 1, "r1")


def test_criticality_rules():
    assert make("exposed_rebar", 10).critical
    assert make("spalling", F.SPALL_CRITICAL_AREA_PX).critical
    assert not make("spalling", F.SPALL_CRITICAL_AREA_PX - 1).critical
    assert not make("crack", 10**7).critical
    assert make().state == "pending_review" and make("crack").state == "draft"


def test_agent_cannot_decide_or_downgrade():
    f = make()
    d = f.to_dict()
    for decision in ("approve", "reject"):
        with pytest.raises(F.ForbiddenTransition):
            F.decide(d, decision, actor="agent")
    assert d["state"] == "pending_review"
    # New, smaller evidence from a finer look must not make a critical finding non-critical.
    big = make("spalling", F.SPALL_CRITICAL_AREA_PX * 2)
    big.agent_update(confidence=0.5, measure={"area_px": 10}, box=[0, 0, 1, 1], step=9, level=0)
    assert big.critical and big.state == "pending_review"
    # A finer look that reveals a larger spall escalates a draft to pending review.
    small = make("spalling", 100)
    small.agent_update(confidence=0.9, measure={"area_px": F.SPALL_CRITICAL_AREA_PX}, box=[0, 0, 1, 1], step=9, level=0)
    assert small.critical and small.state == "pending_review"



def test_evidence_trail():
    """`step` keeps the look that found it; the box spans every look; measure and level are the finest look's."""
    f = make("crack", 900)  # step 3, L1, box [0, 0, 10, 10], confidence 0.8
    f.agent_update(confidence=0.6, measure={"area_px": 400}, box=[6, 4, 10, 10], step=4, level=0, crop_jpeg_b64="fine")
    assert f.step == 3 and f.steps == [3, 4] and f.box == [0, 0, 16, 14]
    assert f.level == 0 and f.measure == {"area_px": 400} and f.confidence == 0.8 and f.crop_jpeg_b64 == "fine"
    # A later, coarser look joins the trail and the box but does not replace the finer measurement.
    f.agent_update(confidence=0.95, measure={"area_px": 9000}, box=[0, 20, 4, 4], step=7, level=2, crop_jpeg_b64="coarse")
    assert f.steps == [3, 4, 7] and f.box == [0, 0, 16, 24] and f.confidence == 0.95
    assert f.level == 0 and f.measure == {"area_px": 400} and f.crop_jpeg_b64 == "fine"
    (wf,) = F.work_order("j_x", [f], [])["findings"]
    assert wf["step"] == 3 and wf["steps"] == [3, 4, 7] and wf["box"] == [0, 0, 16, 24]


def test_criticality_only_rises_along_the_trail():
    s = make("spalling", 100)
    s.agent_update(confidence=0.5, measure={"area_px": F.SPALL_CRITICAL_AREA_PX}, box=[0, 0, 1, 1], step=5, level=2)
    assert s.critical and s.state == "pending_review"  # any look measuring a critical spall sends it to a human
    for step, area in ((6, 10), (7, F.SPALL_CRITICAL_AREA_PX - 1)):
        s.agent_update(confidence=0.5, measure={"area_px": area}, box=[0, 0, 1, 1], step=step, level=0)
        assert s.critical and s.state == "pending_review"

def test_human_decision_once():
    d = make().to_dict()
    F.decide(d, "approve", actor="human")
    assert d["state"] == "approved"
    with pytest.raises(F.NotPending):
        F.decide(d, "reject", actor="human")
    with pytest.raises(F.ReviewError):
        F.decide(make().to_dict(), "downgrade", actor="human")


def test_work_order_signing_and_review():
    wo = F.work_order("j_abc123", [make(), make("crack", fid="f2")], [[1, 2, 1]])
    assert wo["status"] == "blocked" and wo["id"] == "wo_abc123"
    assert all("crop_jpeg_b64" not in f for f in wo["findings"])
    token = F.sign(wo, KEY)
    assert F.verify(token, KEY)["findings"][0]["state"] == "pending_review"
    with pytest.raises(F.NotPending):
        F.apply_review(token, KEY, "f2", "approve", "Ann")  # drafts are not pending review
    token2, wo2 = F.apply_review(token, KEY, "f1", "approve", "  Ann Engineer ")
    assert wo2["status"] == "ready" and wo2["approvals"][0]["reviewer"] == "Ann Engineer"
    assert F.verify(token2, KEY)["findings"][0]["state"] == "approved"
    with pytest.raises(F.NotPending):
        F.apply_review(token2, KEY, "f1", "reject", "Ann")


def test_tampering_expiry_and_reviewer_validation():
    token = F.sign(F.work_order("j_x", [make()], []), KEY)
    body, mac = token.split(".")
    forged = F.sign({"findings": [{"id": "f1", "state": "approved"}]}, b"other-key")
    for bad in (forged, body + "." + mac[:-2] + "AA", "garbage", body[:-4] + "." + mac, ""):
        with pytest.raises(F.SignatureError):
            F.verify(bad, KEY)
    with pytest.raises(F.SignatureError, match="expired"):
        F.verify(F.sign({"findings": []}, KEY, ttl_s=-1), KEY)
    for name in ("", " ", "x" * 65, "evil\nname", "tab\tname"):
        with pytest.raises(F.ReviewError):
            F.apply_review(token, KEY, "f1", "approve", name)
    with pytest.raises(F.ReviewError):
        F.apply_review(token, KEY, "f9", "approve", "Ann")


def test_sign_is_fast_and_small():
    wo = F.work_order("j_x", [make(fid=f"f{i}") for i in range(20)], [[1, 2, 1]] * 60)
    t = time.perf_counter()
    token = F.sign(wo, KEY)
    assert time.perf_counter() - t < 0.05 and len(token) < 20_000
