"""bench.compare_arms on three fake arms cut from the real local micro/macro JSON, with known slow-downs."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from bench import compare_arms

LOCAL = Path(__file__).resolve().parents[1] / "results" / "local-macos-x86"
MICRO, MACRO = (json.loads((LOCAL / f).read_text()) for f in ("micro.json", "macro.json"))
RNG = np.random.default_rng(0)
CROPS = RNG.integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
LABELS = RNG.integers(0, 6, (3, 16, 16), dtype=np.uint8)


def arm(root: Path, name: str, slower: float, usd_per_hour: float, cv2="5.0.0", sha="aa", labels=LABELS,
        crops=CROPS, images=None, parity_sha=None) -> str:
    """Write results/<name>: every op and every job `slower` times the local timings."""
    d = root / name
    d.mkdir()
    micro, macro = copy.deepcopy(MICRO), copy.deepcopy(MACRO)
    for doc in (micro, macro):
        doc["provenance"].update(cv2=cv2, build_info_sha256=sha)
    for row in micro["results"]:
        for k in ("min_ms", "p50_ms", "p95_ms", "p99_ms", "mean_ms"):
            row[k] *= slower
    s = macro["summary"]
    s.update(images_per_s=s["images_per_s"] / slower, wall_s=s["wall_s"] * slower, usd_per_hour=usd_per_hour)
    if images is not None:
        macro["per_image"] = macro["per_image"][:images]
    (d / "micro.json").write_text(json.dumps(micro))
    (d / "macro.json").write_text(json.dumps(macro))
    meta = json.dumps({**micro["provenance"], "build_info_sha256": parity_sha or sha})
    np.savez_compressed(d / "parity.npz", crops=crops, labels=labels, meta=meta)
    return str(d)


def three(tmp_path, **c_kw) -> list[str]:
    b_labels = LABELS.copy()
    b_labels[0, 0, :4] = (b_labels[0, 0, :4] + 1) % 6  # 4 of 768 pixels disagree between A and B
    return [arm(tmp_path, "A", 1.0, 0.339), arm(tmp_path, "B", 1.5, 0.319, labels=b_labels),
            arm(tmp_path, "C", 1.8, 0.357, **c_kw)]


def test_speedups_cost_and_parity(tmp_path):
    rep = compare_arms.build(three(tmp_path))
    assert len(rep["ops"]) == len(MICRO["results"])
    for o in rep["ops"]:
        assert o["cool_gain"] == pytest.approx(1.5, abs=1e-3) and o["hardware_gain"] == pytest.approx(1.2, abs=1e-3)
    assert rep["op_level_geomean"] == pytest.approx({"cool_gain": 1.5, "hardware_gain": 1.2}, abs=1e-3)
    assert rep["end_to_end"] == pytest.approx({"cool_gain": 1.5, "hardware_gain": 1.2, "a_over_c": 1.8}, abs=1e-3)
    a, b, c = (rep["arms"][r] for r in "ABC")
    # arm A is the local run unchanged, so its $/1k must match what bench.macro wrote
    assert a["usd_per_1k_images"] == pytest.approx(MACRO["summary"]["usd_per_1k_images"], abs=1e-5)
    assert b["usd_per_1k_images"] == pytest.approx(0.319 / 3.6 / (MACRO["summary"]["images_per_s"] / 1.5), abs=1e-6)
    assert (a["instance_usd_per_hour"], a["cool_fee_usd_per_hour"]) == (0.319, 0.02)
    assert (c["instance_usd_per_hour"], c["cool_fee_usd_per_hour"]) == (0.357, 0.0)
    par = rep["parity_a_vs_b"]
    assert par["pixel_agreement"] == pytest.approx(1 - 4 / LABELS.size, abs=1e-6)
    assert par["identical_crops"] == 2 and set(par["class_iou"]) == {
        "background", "crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"}


@pytest.mark.parametrize("c_kw, why", [
    ({"cv2": "4.10.0"}, "major versions differ"),
    ({"images": 7}, "macro images differ"),
    ({"crops": CROPS[::-1].copy()}, "different parity crops"),
    ({"parity_sha": "bb"}, "different cv2 builds"),
])
def test_refuses_mismatched_arms(tmp_path, c_kw, why):
    with pytest.raises(SystemExit, match=why):
        compare_arms.main([*three(tmp_path, **c_kw), "--out", str(tmp_path / "r.md"),
                           "--summary", str(tmp_path / "s.json")])
    assert not (tmp_path / "r.md").exists()


def test_markdown_and_summary_render(tmp_path):
    out, summ = tmp_path / "cool_benchmark.md", tmp_path / "cool.json"
    compare_arms.main([*three(tmp_path), "--out", str(out), "--summary", str(summ)])
    md = out.read_text()
    for s in ("# COOL on Graviton", "Expected (docs/PRD.md): 1.2–1.8× op level", "1.2–1.3× end to end",
              "Measured: COOL gain 1.5×, hardware gain 1.2×, A over C 1.8×", "| pyrDown | 1024x768 |",
              "| kleidicv_covered_imgproc | 9.3 % (36.69 ms) |", "## Mask parity, A vs B", "| C | stock wheel, x86 |"):
        assert s in md, s
    table = []
    for line in [*md.splitlines(), ""]:  # every row of a table has as many cells as its header
        if line.startswith("|"):
            table.append(line.count("|"))
        elif table:
            assert len(set(table)) == 1, table
            table = []
    site = json.loads(summ.read_text())
    assert set(site["arms"]) == {"A", "B", "C"} and site["expected"] == {"op_level": [1.2, 1.8], "end_to_end": [1.2, 1.3]}
    assert site["end_to_end"]["cool_gain"] == pytest.approx(1.5, abs=1e-3)
