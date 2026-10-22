import hashlib
import json
import time

import numpy as np
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

def test_the_look_that_made_it_critical_is_recorded():
    """Criticality can come from any look in the trail while `measure` is the finest look's: the work order says
    which look crossed the threshold and with what area, so a 30k px^2 measure on a critical spall is explained."""
    s = make("spalling", 30_000)  # found at step 3, L1: below the threshold
    assert (s.critical_step, s.critical_area_px) == (None, None)
    s.agent_update(confidence=0.5, measure={"area_px": 52_000}, box=[0, 0, 1, 1], step=5, level=2)  # coarser look
    s.agent_update(confidence=0.5, measure={"area_px": 61_000}, box=[0, 0, 1, 1], step=6, level=2)  # already critical
    s.agent_update(confidence=0.5, measure={"area_px": 31_000}, box=[0, 0, 1, 1], step=7, level=0)  # finest look
    assert s.critical and s.measure["area_px"] == 31_000 and s.level == 0
    assert (s.critical_step, s.critical_area_px) == (5, 52_000)
    rebar = make("exposed_rebar", 500)  # critical by class from the look that found it
    assert (rebar.critical_step, rebar.critical_area_px) == (3, 500)
    wo = F.work_order("j_x", [s, rebar, make("crack", fid="f3")], [])
    assert [(f["critical_step"], f["critical_area_px"]) for f in wo["findings"]] == [(5, 52_000), (3, 500), (None, None)]


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


def test_two_decisions_forked_from_one_token_are_visible_siblings():
    """Nothing stops one pre-review token being signed into both an approve and a reject; each order names the
    token it came from (prev_token_sha256), so the fork is visible, and a chained review names its own parent.
    The parent's MAC never appears in the child, so a child cannot be turned back into its parent token."""
    def sha(t):
        return hashlib.sha256(t.encode()).hexdigest()

    two = [make(), make("spalling", F.SPALL_CRITICAL_AREA_PX, fid="f2")]  # both critical
    token = F.sign(F.work_order("j_x", two, []), KEY)
    (ta, a), (tr, r) = (F.apply_review(token, KEY, "f1", d, "Ann") for d in ("approve", "reject"))
    assert F.verify(ta, KEY)["approvals"] == a["approvals"]  # both orders are validly signed
    assert F.verify(tr, KEY)["approvals"] == r["approvals"]
    assert a["approvals"][0]["prev_token_sha256"] == r["approvals"][0]["prev_token_sha256"] == sha(token)
    tc, chained = F.apply_review(ta, KEY, "f2", "approve", "Ann")
    assert [x["prev_token_sha256"] for x in chained["approvals"]] == [sha(token), sha(ta)]
    for parent, child in ((token, ta), (token, tr), (ta, tc)):
        assert parent.split(".")[1] not in F._unb64(child.split(".")[0]).decode()


def test_a_signed_token_has_one_spelling():
    """A lenient base64 decoder reads padding and '+' '/' as '-' '_' to the same MAC bytes, so a re-spelt token
    verified and a second decision forked from one token named a different prev_token_sha256. Only the spelling
    sign() wrote is accepted."""
    token = F.sign(F.work_order("j_x", [make()], []), KEY)
    body, mac = token.split(".")
    for bad in {token + "=", token + "==", body + "." + mac.translate(str.maketrans("-_", "+/"))} - {token}:
        with pytest.raises(F.SignatureError):
            F.verify(bad, KEY)
    assert F.verify(token, KEY)["job"] == "j_x"


def test_tampering_expiry_and_reviewer_validation():
    token = F.sign(F.work_order("j_x", [make()], []), KEY)
    body, mac = token.split(".")
    m = len(mac) // 2  # flip one middle character: overwriting the tail with "AA" was a no-op when the MAC ended in AA
    flipped = mac[:m] + ("B" if mac[m] == "A" else "A") + mac[m + 1:]
    forged = F.sign({"findings": [{"id": "f1", "state": "approved"}]}, b"other-key")
    for bad in (forged, body + "." + flipped, "garbage", body[:-4] + "." + mac, ""):
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


def test_critical_rule_names_the_rule_and_the_measurement():
    assert make("exposed_rebar", 10).critical_rule == "exposed_rebar: always critical"
    assert make("spalling", 48_568).critical_rule == "spalling 48,568 px² >= 40,000 px²"
    assert make("spalling", 100).critical_rule is None and make("crack", 10**7).critical_rule is None
    s = make("spalling", 100)
    s.agent_update(confidence=0.5, measure={"area_px": 52_000}, box=[0, 0, 1, 1], step=5, level=2)
    s.agent_update(confidence=0.5, measure={"area_px": 61_000}, box=[0, 0, 1, 1], step=6, level=2)
    assert s.critical_rule == "spalling 52,000 px² >= 40,000 px²"  # the look that made it critical, as critical_step
    (wf,) = F.work_order("j_x", [s], [])["findings"]
    assert wf["critical_rule"] == s.critical_rule


def test_verification_result_follows_the_latest_check():
    f = make("crack", 900)
    assert (f.verify_iou, f.verified) == (None, None)
    f.agent_update(confidence=0.5, measure={"area_px": 400}, box=[0, 0, 1, 1], step=4, level=0, verify_iou=0.07)
    assert (f.verify_iou, f.verified) == (0.07, False)  # unconfirmed
    f.agent_update(confidence=0.5, measure={"area_px": 400}, box=[0, 0, 1, 1], step=5, level=0)  # no check: kept
    assert (f.verify_iou, f.verified) == (0.07, False)
    g = F.Finding("f2", "crack", 0.8, {"area_px": 10}, [0, 0, 1, 1], 3, 1, "r1", verify_iou=np.float64(0.94))
    assert g.verified is True and type(g.verify_iou) is float
    json.dumps([f.to_dict(), g.to_dict()])  # numpy scalars never reach the JSON


def test_work_order_lists_what_was_not_assessed():
    na = [{"step": 3, "box": [0, 0, 512, 512], "reason": "glare", "action": "re-shoot"}]
    assert F.work_order("j_x", [], [], na)["not_assessed"] == na
    assert F.work_order("j_x", [], [])["not_assessed"] == []


def test_an_order_with_areas_not_assessed_is_incomplete_not_ready():
    """ready only when nothing is left for a person: no critical finding pending and no area not assessed."""
    na = [{"step": 0, "box": [0, 0, 64, 64], "reason": "no candidate region was proposed", "action": "check-by-eye"}]
    assert F.work_order("j_x", [], [], na)["status"] == "incomplete"
    wo = F.work_order("j_x", [make()], [[1, 2, 1]], na)
    assert wo["status"] == "blocked"  # a pending critical finding comes first
    _, wo2 = F.apply_review(F.sign(wo, KEY), KEY, "f1", "approve", "Ann")
    assert wo2["status"] == "incomplete"


# ---- checked by eye: a person resolves a not_assessed entry (docs/api-contract.md, "Checked by eye") ----------------
NA = [{"step": 0, "box": [0, 0, 64, 64], "reason": "no candidate region was proposed", "action": "check-by-eye"},
      {"step": 7, "box": [10, 10, 30, 20], "reason": "spalling at L2, confidence 0.55 under the floor 0.6: not a critical "
       "finding; no confirm look", "action": "check-by-eye", "kind": "needs-a-look", "cls": "spalling", "confidence": 0.55}]


def na_order(critical=True):
    return F.work_order("j_x", [make()] if critical else [], [[1, 2, 1]], json.loads(json.dumps(NA)))


def test_a_person_resolves_each_not_assessed_entry():
    """ready only when every critical finding is decided AND every not-assessed entry is resolved; each resolution is
    a signed, chained audit entry like a decision, and a defect found by eye is who, when and a note, nothing more."""
    def sha(t):
        return hashlib.sha256(t.encode()).hexdigest()

    wo = na_order()
    assert wo["status"] == "blocked" and all("resolution" not in n for n in wo["not_assessed"])  # born unresolved
    t0 = F.sign(wo, KEY)
    t1, wo = F.apply_review(t0, KEY, "f1", "approve", "Ann")
    assert wo["status"] == "incomplete"
    t2, wo = F.apply_resolution(t1, KEY, 0, "nothing-found", " Ann Engineer ")
    assert wo["status"] == "incomplete"  # entry 1 still waits
    t3, wo = F.apply_resolution(t2, KEY, 1, "defect-found", "Ann Engineer", note="  spall at the\nbearing ")
    assert wo["status"] == "ready" and F.verify(t3, KEY) == wo
    r0, r1 = (n["resolution"] for n in wo["not_assessed"])
    assert (r0["decision"], r0["reviewer"], r0["note"]) == ("nothing-found", "Ann Engineer", "")
    assert set(r1) == {"decision", "reviewer", "at", "note"} and r1["note"] == "spall at the bearing"
    assert {k: v for k, v in wo["not_assessed"][1].items() if k != "resolution"} == NA[1]  # the entry never changes
    assert [f["id"] for f in wo["findings"]] == ["f1"]  # a defect found by eye is not an agent finding
    assert [(a.get("finding_id"), a.get("not_assessed"), a["decision"]) for a in wo["approvals"]] == [
        ("f1", None, "approve"), (None, 0, "nothing-found"), (None, 1, "defect-found")]
    assert [a["prev_token_sha256"] for a in wo["approvals"]] == [sha(t0), sha(t1), sha(t2)]  # one chain
    # two resolutions forked from one token are visible siblings, as two decisions are
    (_, a), (_, b) = (F.apply_resolution(t1, KEY, 0, d, "Ann") for d in ("nothing-found", "defect-found"))
    assert a["approvals"][-1]["prev_token_sha256"] == b["approvals"][-1]["prev_token_sha256"] == sha(t1)


def test_only_a_person_resolves_a_not_assessed_entry():
    wo = na_order(critical=False)
    for decision in ("nothing-found", "defect-found"):
        with pytest.raises(F.ForbiddenTransition):
            F.resolve_entry(wo, 0, decision, actor="agent", reviewer="agent")
    assert wo["status"] == "incomplete" and all("resolution" not in n for n in wo["not_assessed"])


def test_resolution_refusals():
    """An index the order does not list, an entry already resolved, a decision of the other kind, a bad name or note,
    and a tampered, forged, expired or replayed token are each refused, and nothing is signed."""
    token = F.sign(na_order(critical=False), KEY)
    for index in (-1, 2, 99, "0", True, None, 0.0):
        with pytest.raises(F.ReviewError, match="no not_assessed entry"):
            F.apply_resolution(token, KEY, index, "nothing-found", "Ann")
    for decision in ("approve", "reject", "close", "checked", ""):
        with pytest.raises(F.ReviewError):
            F.apply_resolution(token, KEY, 0, decision, "Ann")
    for name in ("", " ", "x" * 65, "evil\nname"):
        with pytest.raises(F.ReviewError, match="reviewer"):
            F.apply_resolution(token, KEY, 0, "nothing-found", name)
    for note in ("x" * 201, "bell\x07", "nul\x00"):
        with pytest.raises(F.ReviewError, match="note"):
            F.apply_resolution(token, KEY, 0, "nothing-found", "Ann", note=note)
    body, mac = token.split(".")
    flipped = mac[:5] + ("B" if mac[5] == "A" else "A") + mac[6:]
    for bad in (body + "." + flipped, F.sign(na_order(critical=False), b"other-key" * 4), "garbage",
                F.sign(na_order(critical=False), KEY, ttl_s=-1)):
        with pytest.raises(F.SignatureError):
            F.apply_resolution(bad, KEY, 0, "nothing-found", "Ann")
    done, wo = F.apply_resolution(token, KEY, 0, "nothing-found", "Ann")
    with pytest.raises(F.NotPending, match="already resolved: nothing-found by Ann"):  # the returned token, replayed
        F.apply_resolution(done, KEY, 0, "defect-found", "Bob")
    assert wo["not_assessed"][0]["resolution"]["decision"] == "nothing-found"
    with pytest.raises(F.ReviewError):  # an entry is not a finding
        F.apply_review(token, KEY, "na0", "approve", "Ann")


def test_an_order_without_resolutions_keeps_its_old_status():
    """A work order written before resolutions existed (1.1.1: no entry resolved, every approval a finding_id)
    verifies and computes the status the old rule gave; the agent's orders never carry a resolution."""
    def old_rule(wo):
        if any(f["state"] == "pending_review" for f in wo["findings"]):
            return "blocked"
        return "incomplete" if wo["not_assessed"] else "ready"

    for crit in (True, False):
        for na in ([], NA[:1], NA):
            wo = F.work_order("j_x", [make()] if crit else [], [], json.loads(json.dumps(na)))
            assert F.order_status(wo) == old_rule(wo) == wo["status"]
            token = F.sign(wo, KEY)
            if crit:
                token, wo = F.apply_review(token, KEY, "f1", "reject", "Ann")
                assert F.order_status(wo) == old_rule(wo) == wo["status"]
            assert F.verify(token, KEY) == wo and "resolution" not in json.dumps(wo)


def test_only_the_human_surfaces_call_the_human_transitions():
    """decide/apply_review and resolve_entry/apply_resolution are named only in findings.py and the two human surfaces
    (web.py's /api/review, cli.py's `drishyam review`): no agent, planner, MCP, batch or cascade module reaches them."""
    import ast
    from pathlib import Path

    human = {"decide", "apply_review", "resolve_entry", "apply_resolution"}
    named = {}
    for path in sorted(Path(F.__file__).parent.rglob("*.py")):
        tree = ast.parse(path.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        names |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        names |= {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        if names & human:
            named[path.name] = names & human
    assert set(named) == {"findings.py", "web.py", "cli.py"}, named
    assert named["web.py"] >= {"apply_review", "apply_resolution"} and named["cli.py"] >= {"decide", "resolve_entry"}
