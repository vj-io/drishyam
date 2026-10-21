"""bench/opencv5_engines.py on bench/repro's tiny Shape -> Resize model (no weights needed), and docs/bench.md's
"OpenCV 5 DNN engines for v1" section is the committed JSON in results/local-macos-x86/.
`python -m tests.test_opencv5_engines` prints the table rows."""
import json
from pathlib import Path

import pytest

pytest.importorskip("onnx")
from bench import opencv5_engines as eng
from bench.repro.classic_shape import build

REPO = Path(__file__).resolve().parents[1]
RUNS = ("opencv5_engines.json", "opencv5_engines_repeat.json")
SHAPE = "!isDynamicShape in function 'parseShape'"
LABEL = {"drishyam_dacl10k.onnx": "FP32 (the deployed file)", "drishyam_dacl10k.fp16.onnx": "FP16",
         "drishyam_dacl10k.onnx static": "FP32, static copy", "drishyam_dacl10k.fp16.onnx static": "FP16, static copy"}


def test_engines_on_the_tiny_repro(tmp_path):
    model = build(str(tmp_path / "dynamic_hw.onnx"), [1, 3, "h", "w"])
    doc = eng.run("tiny", [model], repeats=2, crops=1, out_dir=str(tmp_path), work=str(tmp_path / "work"), sizes=(64,))
    assert json.loads((tmp_path / "tiny" / "opencv5_engines.json").read_text()) == doc
    rows = {(r["file"], r["engine"]): r for r in doc["rows"]}
    assert set(rows) == {(f, e) for f in ("dynamic_hw.onnx", "dynamic_hw.onnx static") for e in ("new", "classic")}
    dyn, static = rows["dynamic_hw.onnx", "classic"], rows["dynamic_hw.onnx static", "classic"]
    assert not dyn["loads"] and SHAPE in dyn["error"]
    assert any("dynamic 'zero' shapes are not supported" in ln for ln in dyn["log"])  # OpenCV's log, from fd 2
    assert static["runs"] and static["vs_reference"]["max_dprob"] < 1e-6 and static["vs_other_engine"]["argmax_agreement"] == 1
    assert rows["dynamic_hw.onnx", "new"]["vs_reference"] == "reference"
    for r in (r for r in doc["rows"] if r["runs"]):
        assert {"forward_threads1", f"forward_threads{eng.ALL}"} <= r.keys() and r["forward_threads1"]["n"] == 2
    assert "null_forward_threads1" in rows["dynamic_hw.onnx", "new"]
    g = doc["files"]["dynamic_hw.onnx"]["graph"]
    assert g["inputs"] == {"x": [1, 3, "h", "w"]} and [s["chain"] for s in g["shape_nodes"]] == [
        "Shape -> Gather -> Unsqueeze -> Concat -> Resize"]
    assert {"cv2_baseline", "cv2_dispatch", "cv2_custom_hal", "cv2_parallel_framework"} <= doc["provenance"].keys()
    assert doc["onnx_runtime_in_build"] is not None


def test_missing_weights_stop_before_any_timing(tmp_path):
    with pytest.raises(SystemExit, match="release assets"):
        eng.run("x", [str(tmp_path / "drishyam_dacl10k.onnx")], out_dir=str(tmp_path))
    assert not (tmp_path / "x").exists()


def load(name: str) -> dict:
    return json.loads((REPO / "results" / "local-macos-x86" / name).read_text())


def ms(t: dict) -> str:
    return " / ".join(f"{t[k]:.1f}" for k in ("p50_ms", "p95_ms"))


def num(v: float) -> str:
    """0, 2.1e-5, 0.018: the max |dprob| as the section writes it."""
    return "0" if v == 0 else f"{v:.1e}".replace("e-0", "e-") if v < 1e-3 else f"{v:.2g}"


def dp(v) -> str:
    return "(reference)" if v == "reference" else "" if v is None else f"{num(v['max_dprob'])}, {v['argmax_agreement']:.6f}"


def table_rows():
    doc = load(RUNS[0])
    n = doc["config"]["all_threads"]
    for r in doc["rows"]:
        status = "runs" if r["runs"] else "loads, forward fails (b)" if r["loads"] else "fails to load (a)"
        t = lambda k: ms(r[k]) if k in r else ""  # noqa: B023
        yield (f"| {LABEL[r['file']]} | `ENGINE_{r['engine'].upper()}` | {r['size']} | {status} | {t('forward_threads1')} | "
               f"{t(f'forward_threads{n}')} | {dp(r.get('vs_reference'))} | {dp(r.get('vs_other_engine'))} |")
        if "null_forward_threads1" in r:
            yield (f"| FP32, the same net in a second timing slot (null) | `ENGINE_NEW` | {r['size']} | | "
                   f"{t('null_forward_threads1')} | {t(f'null_forward_threads{n}')} | | |")


def p50(doc: dict, file: str, engine: str, size: int, key: str) -> float:
    return next(r for r in doc["rows"] if (r["file"], r["engine"], r["size"]) == (file, engine, size))[key]["p50_ms"]


def ratios(doc: dict, a: tuple, b: tuple, key_a: str = "forward_threads{}", key_b: str = "forward_threads{}") -> list:
    """p50 of a over b at 512 and 256, on 1 thread and then all threads: [512/1, 256/1, 512/N, 256/N]."""
    n = doc["config"]["all_threads"]
    return [p50(doc, *a, s, key_a.format(t)) / p50(doc, *b, s, key_b.format(t)) for t in (1, n) for s in (512, 256)]


def section() -> str:
    return " ".join((REPO / "docs" / "bench.md").read_text().split("## OpenCV 5 DNN engines for v1\n")[1]
                    .split("\n## ")[0].split())


def test_engines_section_is_the_json():
    text = (REPO / "docs" / "bench.md").read_text().split("## OpenCV 5 DNN engines for v1\n")[1].split("\n## ")[0]
    rows = list(table_rows())
    assert len(rows) == 18 and [r for r in rows if r not in text] == []
    a, b = (load(f) for f in RUNS)
    fp32, fp16 = "drishyam_dacl10k.onnx", "drishyam_dacl10k.fp16.onnx"
    strip = lambda d: [{k: v for k, v in r.items() if "forward" not in k and k != "log"} for r in d["rows"]]
    assert strip(a) == strip(b)  # loads, errors and parity reproduced exactly; only the timings differ
    for d in (a, b):
        errs = {(r["file"], r["engine"]): r for r in d["rows"] if not r["runs"]}
        assert {k for k, r in errs.items() if SHAPE in r.get("error", "")} == {(fp32, "classic"), (fp16, "classic")}
        assert set(errs) == {(fp32, "classic"), (fp16, "classic"), (fp16 + " static", "classic")}
    rel = lambda v: ", ".join(f"x{x:.2f}" for x in v)
    pct = lambda v: f"{100 * max(abs(x - 1) for x in v):.0f} %"
    classic = [ratios(d, (fp32 + " static", "classic"), (fp32, "new")) for d in (a, b)]
    assert all(x < 1 for c in classic for x in c[:2]) and all(x > 1 for c in classic for x in c[2:])
    half = [ratios(d, (fp16, "new"), (fp32, "new")) for d in (a, b)]
    assert all(x > 1 for h in half for x in h)
    static_new = [ratios(d, (fp32 + " static", "new"), (fp32, "new")) for d in (a, b)]
    null = [ratios(d, (fp32, "new"), (fp32, "new"), "null_forward_threads{}") for d in (a, b)]
    ref = {r["size"]: r for r in a["rows"] if (r["file"], r["engine"]) == (fp32 + " static", "classic")}
    pa, pb = (d["provenance"] for d in (a, b))
    want = [f"1-minute load average {pa['load_avg'][0]} at the start and {a['load_avg_end'][0]} at the end",
            f"from {pb['load_avg'][0]} to {b['load_avg_end'][0]}",
            f"{a['config']['crops']} synthetic crops", f"{a['config']['repeats']} rounds",
            f"on 1 and {a['config']['all_threads']} threads",
            f"max |dprob| {num(ref[512]['vs_reference']['max_dprob'])} at 512 and {num(ref[256]['vs_reference']['max_dprob'])} at 256",
            f"at one thread it took {rel(classic[0][:2])} of the deployed net's p50 at 512 and 256 ({rel(classic[1][:2])} in the repeat)",
            f"at {a['config']['all_threads']} threads {rel(classic[0][2:])} ({rel(classic[1][2:])})",
            f"the null slot's p50 was within {pct(null[0])} of the deployed net's ({pct(null[1])} in the repeat)",
            f"within {pct(static_new[0])} ({pct(static_new[1])})",
            (f"FP16 file on `ENGINE_NEW` took {rel(half[0])} of the FP32 file's p50 at 512 and 256 on one thread, then "
             f"on {a['config']['all_threads']} ({rel(half[1])} in the repeat)"),
            f"`{a['onnx_runtime_in_build']}`", f"`cv2_custom_hal` {pa['cv2_custom_hal']}", pa["cv2_dispatch"]]
    assert [w for w in want if w not in section()] == []


if __name__ == "__main__":
    print(*table_rows(), sep="\n")
