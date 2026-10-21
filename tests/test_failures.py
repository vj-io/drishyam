"""The failure gallery's rules say what is implemented, and each case says why it still slipped past (evals/failures.py)."""
import cv2
import numpy as np

from evals import failures

QUIET = {"preds": [], "trace": []}


def gallery(tmp_path, rec, cfg):
    path = tmp_path / f"{rec['id']}.jpg"
    cv2.imwrite(str(path), np.full((1536, 2048, 3), 150, np.uint8))
    res = {"configs": {"drishyam@0.3": QUIET, "drishyam@1.0": QUIET, "tiling@L0": QUIET, **cfg}}
    return failures.build_gallery([{**rec, "path": str(path)}], [res], tmp_path / "pub", log=lambda *_: None)


def look(step, level, window, ok=True, reason=None):
    return {"step": step, "action": "gaze", "cause": "survey", "x": window[0] + window[2] // 2, "y": window[1] + window[3] // 2,
            "level": level, "window": window, "q_ok": ok, "q_reason": reason, "sharpness": 40.0, "model": "accurate", "governor": None}


def test_rules_name_what_is_implemented():
    rule = {title: text for title, _, text in failures.CATEGORIES}
    for title, where in (("Formwork line read as crack", "drop_straight_lines"), ("Glare caught by the quality gate", "not_assessed"),
                         ("Rebar, spall and stain confusion", "review on ambiguity")):
        assert rule[title].startswith("Implemented") and where in rule[title], title
        assert "Not implemented:" in rule[title]  # and the part still a proposal is named, not dropped


def test_a_formwork_crack_says_which_look_kept_it(tmp_path):
    rec = {"id": "s0001", "boxes": [], "distractors": [{"kind": "formwork_line", "box": [1200, 400, 640, 48]}]}
    pred = {"cls": "crack", "box": [1216, 412, 624, 40], "area": 5744, "conf": 0.88, "step": 1, "level": 2, "region": "s0"}
    trace = [look(1, 2, [0, -256, 2048, 2048]), look(2, 1, [400, 47, 1024, 1024])]  # step 2 misses the line's centre
    (row,) = gallery(tmp_path, rec, {"drishyam@0.3": {"preds": [pred], "trace": trace}})
    assert row["category"] == "Formwork line read as crack"
    assert "kept it at the look that drafted it (step 1, L2, where a residual counts 4x in L0 px)" in row["what_happened"]
    assert "and no later look covered it." in row["what_happened"]  # step 2's window misses the line's centre


def test_a_formwork_crack_counts_only_the_looks_after_its_draft(tmp_path):
    """The look that drafted the crack, and any look before it, are not 'a later look'."""
    rec = {"id": "s0004", "boxes": [], "distractors": [{"kind": "formwork_line", "box": [1200, 400, 640, 48]}]}
    pred = {"cls": "crack", "box": [1216, 412, 624, 40], "area": 5744, "conf": 0.88, "step": 3, "level": 1, "region": "s0"}
    window = [1016, -80, 1024, 1024]
    early = [look(1, 2, [0, -256, 2048, 2048]), look(2, 0, [1272, 176, 512, 512]), look(3, 1, window)]
    (row,) = gallery(tmp_path, rec, {"drishyam@0.3": {"preds": [pred], "trace": early}})
    assert "(step 3, L1, where a residual counts 2x in L0 px), and no later look covered it." in row["what_happened"]
    assert "a later full-resolution look" not in row["what_happened"]
    late = early + [look(4, 0, [1272, 176, 512, 512])]
    (row,) = gallery(tmp_path, rec, {"drishyam@0.3": {"preds": [pred], "trace": late}})
    assert "; a later full-resolution look covered it, but a later look never withdraws" in row["what_happened"]


def test_a_formwork_crack_drafted_at_full_resolution_says_the_test_ran_there(tmp_path):
    rec = {"id": "s0005", "boxes": [], "distractors": [{"kind": "formwork_line", "box": [1200, 400, 640, 48]}]}
    pred = {"cls": "crack", "box": [1216, 412, 624, 40], "area": 5744, "conf": 0.88, "step": 3, "level": 0, "region": "s0"}
    trace = [look(1, 2, [0, -256, 2048, 2048]), look(3, 0, [1272, 176, 512, 512])]
    (row,) = gallery(tmp_path, rec, {"drishyam@0.3": {"preds": [pred], "trace": trace}})
    assert "ran at full resolution on the look that drafted it (step 3, L0) and kept it" in row["what_happened"]
    assert "later" not in row["what_happened"]


def test_a_glare_case_says_the_window_is_listed_not_assessed(tmp_path):
    rec = {"id": "s0002", "boxes": [], "distractors": [{"kind": "glare", "box": [400, 800, 200, 200]}]}
    trace = [look(1, 2, [0, -256, 2048, 2048]), look(2, 0, [313, 768, 512, 512], ok=False, reason="glare")]
    (row,) = gallery(tmp_path, rec, {"drishyam@0.3": {"preds": [], "trace": trace}})
    assert "lists that window as not assessed (action re-shoot)" in row["what_happened"] and "silently" not in row["what_happened"]


def test_a_spall_reported_as_rebar_says_why_review_on_ambiguity_cannot_fire(tmp_path):
    rec = {"id": "s0003", "boxes": [{"cls": "spalling", "box": [560, 520, 120, 200]}], "distractors": []}
    rebar = {"cls": "exposed_rebar", "box": [570, 530, 100, 180], "area": 9000, "conf": 0.93, "step": 4, "level": 0, "region": "r1"}
    (row,) = gallery(tmp_path, rec, {"drishyam@1.0": {"preds": [rebar], "trace": [look(4, 0, [342, 428, 512, 512])]}})
    assert row["category"] == "Rebar, spall and stain confusion"
    assert "cannot flag it" in row["what_happened"] and "spalling is never absorbed into exposed rebar" in row["what_happened"]
