"""Since 2026-10-21 the deployed demo runs v1, the dacl10k-trained segmenter (docs/aws.md), and its held-out results
are published (results/heldout_models.json, public/results/{codebrim,synth}_dacl10k/). These tests fail if docs/,
public/*.html, public/*.js, public/js/*.js, README.md or models/README.md say, in a sentence about the live or
deployed demo, that it runs the classical engine (classical-v1), or that the trained segmenter's results are
unpublished or that no trained model is live. They also fail if a line the 2026-10-21 re-grade found stale comes back,
or if the Lambda build or cold start the docs quote drifts from results/lambda-cv2-build.json and
video/live/aws-smoke.log. `python -m tests.test_live_state_doc` prints every sentence it flags."""
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# docs/PRD.md is the original plan, marked so at its top and not kept current, so it is not checked.
FILES = sorted({*REPO.glob("docs/**/*.md"), *REPO.glob("public/*.html"), *REPO.glob("public/*.js"),
                *REPO.glob("public/js/*.js"), REPO / "README.md", REPO / "models/README.md"} - {REPO / "docs/PRD.md"})

LIVE = re.compile(r"\b(?:live|deployed|demo runs|demo engine)\b|/api/health", re.IGNORECASE)
CLASSICAL = re.compile(r"classical-v1|classical (?:engine|segmenter)|engine is classical", re.IGNORECASE)
# A present-tense claim that something runs the classical engine, whatever else the sentence names.
RUNS_CLASSICAL = re.compile(r"(?:runs|uses|reports|is running)\s+(?:the\s+|segmenter\s+|`)*(?:classical-v1|classical (?:engine|segmenter))"
                            r"|segmenter\s+`?classical-v1"
                            r"|classical (?:engine|segmenter),?\s+(?:that\s+|which\s+)?the (?:live |deployed )?(?:demo|site|function) (?:runs|uses)"
                            r"|(?:demo|live|deployed) (?:engine|segmenter) is (?:the )?classical", re.IGNORECASE)
# A sentence that names v1 next to the classical engine compares the two; one that names only the classical engine
# beside the live demo is a claim about what runs, unless it says that was before the switch or is the fallback.
V1 = re.compile(r"\bv1\b|onnx:drishyam_dacl10k|dacl10k-trained", re.IGNORECASE)
NOT_NOW = re.compile(r"until 2026-10-21|before 2026-10-21|on 2026-10-1\d|of 2026-10-1\d|\(2026-10-1\d|fallback|"
                     r"without `?DRISHYAM_ONNX|recorded on the classical|predates", re.IGNORECASE)
UNPUBLISHED = re.compile(r"not (?:yet |been )?published|is the next step|next step is|still to be added|"
                         r"none of these models is the live|no trained model (?:is|runs) live|"
                         r"live demo does not use (?:it|v1|the trained)", re.IGNORECASE)
TRAINED = re.compile(r"\b(?:trained|dacl10k|v1|v2[abc]|models?)\b", re.IGNORECASE)

# The sentences that were true before the switch and false after it (commit 3fd7a49): the rules must flag each one.
OLD = [
    "The live demo uses no OpenCV 5-only API (segmenter classical-v1).",
    ("Its weight files are not committed to this repository, and the live demo does not use it: the deployed "
     "`/api/health` reports segmenter `classical-v1`."),
    ("This report makes no time-saving claim: on real photos Drishyam finds about 0.2 of the labelled defects with the "
     "classical engine the demo runs, and about half with the trained v1 segmenter (section 8)."),
    ("The demo engine is classical (evidence maps, no weights); a dacl10k ONNX model runs on OpenCV 5 DNN `ENGINE_NEW` "
     "with the same interface."),
    "None of these models is the live segmenter.",
    ("On dev only, the recommendation is v1 for the live engine, which runs the classical engine today: it is the one "
     "model with no measurable loss against any other."),
    ("Here the classical segmenter, tuned on synthetic images, limits recall more than where the agent looks; a "
     "trained model is the next step."),
    ("The trained segmenter's summaries (results/synth_dacl10k, results/codebrim_dacl10k) are not published yet; pass "
     "them as renderBench's `model` once they are."),
    "The deployed site uses the classical engine.",  # and two phrasings no old file had
    "Live, the agent runs the classical engine, which tops out near 0.2 recall on real photos.",
]
# Sentences that name both and are true now: the rules must pass each one.
NOW = [
    ("The classical engine (no weights; the deployed demo until 2026-10-21, and every policy comparison) does not let "
     "the agent beat the simple baselines."),
    "Without `DRISHYAM_ONNX` the classical evidence-map engine (`classical-v1`, no weights) runs and needs no downloads.",
    "The live demo runs v1 since 2026-10-21; the classical engine is the fallback without its weights.",
    ("On dev only, the recommendation is v1 for the live engine, which runs it since 2026-10-21 (deployed commit "
     "02219a2): it is the one model with no measurable loss against any other, at 2.00 times the classical engine's "
     "modelled ms per CODEBRIM job on this Mac (CI 1.89-2.11)."),
    ("On the Lambda the only numbers are the smoke runs' step p95s: 67.7 to 88.0 ms with the classical engine and 0.56 "
     "to 1.14 s with v1, the deployed engine."),
]


def stale(sentence: str) -> str | None:
    """Why `sentence` misstates the live state, or None."""
    if LIVE.search(sentence) and not NOT_NOW.search(sentence) and (
            RUNS_CLASSICAL.search(sentence) or CLASSICAL.search(sentence) and not V1.search(sentence)):
        return "says the live or deployed demo runs the classical engine"
    if UNPUBLISHED.search(sentence) and TRAINED.search(sentence):
        return "says the trained segmenter's results are unpublished, or that no trained model is live"
    return None


def sentences(path: Path) -> list[str]:
    text = path.read_text()
    if path.suffix == ".html":
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    # a list item, table row or heading is its own sentence; in JS a statement ends at ";" too
    ends = r"(?<=[.!?;])\s+" if path.suffix == ".js" else r"(?<=[.!?])\s+"
    parts = re.split(rf"{ends}|\n\s*\n|\n(?=\s*(?:[-*]|\d+\.|\||#+|//)\s)", text)
    return [" ".join(p.split()) for p in parts if p.strip()]


def offenders() -> list[str]:
    return [f"{p.relative_to(REPO)}: {why}: {s[:300]}" for p in FILES for s in sentences(p) if (why := stale(s))]


def test_the_rules_flag_the_old_sentences_and_pass_the_true_ones():
    assert [s for s in OLD if not stale(s)] == []
    assert [s for s in NOW if stale(s)] == []


def test_no_doc_or_site_text_says_the_live_demo_runs_the_classical_engine():
    assert len(FILES) > 15 and all(p.exists() for p in FILES)
    assert offenders() == []


def test_the_sites_static_lines_quote_v1s_published_numbers():
    """The hero's and the exposed-rebar sample's line quote v1's synthetic exposed-rebar recall, and the FAQ its chance
    score on CODEBRIM, both at the 30 % cap (policies["drishyam"][3]: evals.harness.BUDGETS order)."""
    syn, cb = (json.loads((REPO / f"public/results/{d}_dacl10k/summary.json").read_text()) for d in ("synth", "codebrim"))
    rebar = f"finds {syn['per_class']['drishyam@0.3']['exposed_rebar']:.2f} of the exposed rebar"
    chance = f"boxes at random places already score {cb['policies']['drishyam'][3]['chance_recall']:.2f}"
    page = html.unescape((REPO / "public/index.html").read_text())
    samples = json.loads((REPO / "public/samples/index.json").read_text())
    (note,) = [s["note"] for s in samples if s["title"] == "Exposed rebar"]
    assert rebar in page and rebar in note and chance in page


# Lines the 2026-10-21 re-grade of v1.1.0 found stale against the live f12f50f build: none may come back. Each phrase
# is matched case-insensitively in FILES and the architecture figure's sources, HTML and SVG with tags stripped.
STALE_CLAIMS = {
    "about 3 s": "the cold start: 0.86 to 6.27 s to the first byte over the v1 smoke runs (docs/aws.md, check 3)",
    "build information is not": "GET /api/health returns the Lambda's cv2_build (results/lambda-cv2-build.json)",
    "empty gaze path": "the wall's work order lists the whole photo as not_assessed, check-by-eye",
    "halted after 2 steps in the": "the f12f50f smoke's KILL halted after 3 steps",
    "zoom until": "v1 zooms in where it is unsure, no finer than L1 or L2 on these photos",
    "l0 · re-segment": "v1 verifies one level finer, never at L0 on a 2048- or 4608-px photo",
}
# A deploy older than the live one is named only in the history that records it (and 03af85b as the model's install
# commit in models/README.md), never as the current build.
OLD_COMMITS = {"02219a2": {"docs/aws.md"}, "03af85b": {"models/README.md"}}
FIGURES = [REPO / "docs/architecture.mmd", REPO / "docs/architecture.svg"]
# The lines as the release had them: the check must flag each one.
OLD_CLAIMS = [
    "That one starts cold (about 3 s with v1), a short job can finish first.",
    "([results/linux-aarch64/micro.json](results/linux-aarch64/micro.json); the Lambda's own build information is not recorded).",
    "Its work order shows an empty gaze path and no finding, which a reader could take for a clean wall.",
    "sent once step 2 has streamed: halted after 2 steps in the 02219a2 smoke run, section 5",
    "2. Zoom until it’s sure",
    "VERIFY L0 · RE-SEGMENT",
]


def flat_text(path: Path) -> str:
    text = path.read_text()
    if path.suffix in (".html", ".svg"):
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return " ".join(text.split())


def stale_claims(text: str) -> list[str]:
    return [p for p in STALE_CLAIMS if p in text.lower()]


def test_the_stale_claims_the_regrade_found_stay_gone():
    assert [s for s in OLD_CLAIMS if not stale_claims(" ".join(s.split()))] == []
    found = [(p.relative_to(REPO).as_posix(), c) for p in FILES + FIGURES for c in stale_claims(flat_text(p))]
    assert found == []
    named = [(p.relative_to(REPO).as_posix(), c) for p in FILES + FIGURES for c, ok in OLD_COMMITS.items()
             if c in p.read_text() and p.relative_to(REPO).as_posix() not in ok]
    assert named == []


def test_the_lambda_build_and_cold_start_are_the_recorded_ones():
    """README.md and docs/report.md quote the Lambda's own cv2_build as results/lambda-cv2-build.json holds it; docs/aws.md
    and public/js/api.js quote the range of the first /api/health's time to first byte over the v1 smoke runs in
    video/live/aws-smoke.log."""
    rec = json.loads((REPO / "results/lambda-cv2-build.json").read_text())["health"]
    b = rec["cv2_build"]
    assert rec["host"] == "aws-lambda" and rec["arch"] == "aarch64" and rec["segmenter"] == "onnx:drishyam_dacl10k.onnx"
    readme, report = (flat_text(REPO / f) for f in ("README.md", "docs/report.md"))
    for name, text, keys in (("README.md", readme, ("Custom HAL", "baseline", "dispatch", "parallel framework")),
                             ("docs/report.md", report, ("hal", "cpu_baseline", "cpu_dispatch", "parallel"))):
        want = [f'{k} "{v}"' for k, v in zip(keys, (b["hal"], b["cpu_baseline"], b["cpu_dispatch"], b["parallel"]), strict=True)]
        assert [w for w in want if w not in text] == [], name
    seg, firsts = None, []
    for line in (REPO / "video/live/aws-smoke.log").read_text().splitlines():
        if m := re.search(r'health: .*"segmenter":"([^"]+)"', line):
            seg = m[1]
        if line.startswith("smoke ") and seg == "onnx:drishyam_dacl10k.onnx":
            firsts.append(float(re.search(r"ttfb_first=([\d.]+)s", line)[1]))
    lo, hi = min(firsts), max(firsts)
    aws = flat_text(REPO / "docs/aws.md")
    assert len(firsts) == 11 and "over the eleven v1 smoke runs the first `/api/health`" in aws
    assert f"took {lo:.2f} to {hi:.2f} s to its first byte" in aws
    assert f"({lo:.2f} to {hi:.2f} s to the first byte over the v1 smoke runs)" in flat_text(REPO / "public/js/api.js")


if __name__ == "__main__":
    print(*offenders(), sep="\n")
