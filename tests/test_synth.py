import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from evals import synth

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "public" / "samples"
FULL_SPEC = {
    "crack": 2, "spalling": 1, "exposed_rebar": 1, "corrosion": 1, "efflorescence": 1,
    "distractors": ["graffiti", "formwork_line", "water_stain", "glare", "blur_region"],
}


def _in_bounds(box, w, h):
    x, y, bw, bh = box
    return x >= 0 and y >= 0 and bw >= 1 and bh >= 1 and x + bw <= w and y + bh <= h


def test_deterministic():
    for spec in (None, FULL_SPEC):
        (a, gt_a), (b, gt_b) = (synth.generate(5, 640, 480, spec) for _ in range(2))
        assert a.dtype == np.uint8 and a.shape == (480, 640, 3)
        assert a.tobytes() == b.tobytes() and gt_a == gt_b


def test_hairline_flag_leaves_the_images_unchanged():
    """Recording the hairline flag in the GT must not move a single random draw: these hashes predate the flag."""
    for spec, digest in ((None, "cc42421dae3ed5e1"), (FULL_SPEC, "c662995f7c2cbd4d"), ({"crack": 3}, "2b3ce7b116f067f1")):
        img, gt = synth.generate(5, 640, 480, spec)
        assert hashlib.sha256(img.tobytes()).hexdigest()[:16] == digest
        assert all(isinstance(g["hairline"], bool) for g in gt if g["cls"] == "crack")
        assert all("hairline" not in g for g in gt if g["cls"] != "crack")
    _, gt, _ = synth.render(5, 640, 480, {"crack": 2, "hairline": True, "distractors": ["blur_region"]})
    assert [g["hairline"] for g in gt] == [True, True, True]  # the crack under a blur patch is a hairline too


def test_seeds_differ():
    a, _ = synth.generate(1, 320, 240)
    b, _ = synth.generate(2, 320, 240)
    assert not np.array_equal(a, b)


def test_boxes_inside_image_and_positive():
    for seed in range(8):
        for spec in (None, FULL_SPEC):
            _, gt, extra = synth.render(seed, 640, 480, spec)
            assert all(g["cls"] in synth.CLASSES and _in_bounds(g["box"], 640, 480) for g in gt)
            assert all(e["kind"] in synth.DISTRACTORS and _in_bounds(e["box"], 640, 480) for e in extra)


def test_spec_counts_are_honoured():
    spec = {"crack": 2, "spalling": 1, "exposed_rebar": 1, "corrosion": 1, "efflorescence": 1,
            "distractors": ["graffiti"]}
    _, gt, extra = synth.render(3, 800, 600, spec)
    got = {c: sum(g["cls"] == c for g in gt) for c in synth.CLASSES}
    # the rebar sits inside a spall, which is its own spalling box
    assert got == {"crack": 2, "spalling": 2, "exposed_rebar": 1, "corrosion": 1, "efflorescence": 1}
    assert got == synth.planned_counts(spec)
    assert [e["kind"] for e in extra] == ["graffiti"]
    # distractors never enter the GT list; blur_region's crack does (it still counts)
    _, gt, extra = synth.render(3, 800, 600, {"distractors": ["glare", "blur_region"]})
    assert [g["cls"] for g in gt] == ["crack"] and [e["kind"] for e in extra] == ["blur_region", "glare"]


def test_random_mix_has_one_to_five_defects():
    for seed in range(40):
        spec = synth._random_spec(np.random.default_rng((seed, 3)))
        assert 1 <= sum(spec[c] for c in synth.CLASSES) <= 5 and len(spec["distractors"]) <= 1


def _crack_contrast(seed, hairline, w=800, h=600):
    """p99 darkening inside the crack's box at full size and after a 1/8 INTER_AREA downscale. The background is
    identical for a given seed, so (image without crack - image with crack) isolates the crack exactly."""
    with_crack, gt = synth.generate(seed, w, h, {"crack": 1, "hairline": hairline})
    clean, _ = synth.generate(seed, w, h, {"crack": 0})
    x, y, bw, bh = gt[0]["box"]
    full = (clean.astype(np.float32) - with_crack).mean(2)[y:y + bh, x:x + bw]

    def small(im):
        return cv2.resize(im, (w // 8, h // 8), interpolation=cv2.INTER_AREA).astype(np.float32)

    low = (small(clean) - small(with_crack)).mean(2)
    low = low[max(y // 8 - 1, 0):-(-(y + bh) // 8) + 1, max(x // 8 - 1, 0):-(-(x + bw) // 8) + 1]
    return float(np.percentile(full, 99)), float(np.percentile(low, 99))


def test_hairline_crack_fades_when_downscaled():
    seeds = (1, 2, 4, 6)
    hair = [_crack_contrast(s, True) for s in seeds]
    for before, after in hair:
        assert before >= 20, "hairline must be visible at full resolution"
        assert after <= 0.65 * before, "and much weaker at 1/8 scale"
    thick = [_crack_contrast(s, False) for s in seeds]
    assert np.mean([a / b for b, a in thick]) > np.mean([a / b for b, a in hair]) + 0.15  # wide cracks survive


def test_stratification_gives_every_class_at_least_40_boxes_over_200_images():
    for n, seed, floor in ((200, 7, synth.MIN_PER_CLASS), (60, 1, 12)):  # small n caps the floor at n // 5
        specs = synth.stratify(n, seed)
        assert len(specs) == n
        for cls in synth.CLASSES:
            assert sum(synth.planned_counts(s)[cls] for s in specs) >= floor, (n, cls)
    assert synth.stratify(200, 7) == synth.stratify(200, 7)


def test_build_dataset_writes_images_and_annotations(tmp_path):
    synth.build_dataset(tmp_path, n=3, seed=7, width=320, height=240)
    ann = json.loads((tmp_path / "annotations.json").read_text())
    assert ann["dataset"] == "synthetic-3" and ann["synthetic"] is True and ann["seed"] == 7
    assert [i["id"] for i in ann["images"]] == ["s0001", "s0002", "s0003"]
    for im in ann["images"]:
        assert cv2.imread(str(tmp_path / im["file"])).shape == (im["height"], im["width"], 3)
        assert all(_in_bounds(b["box"], im["width"], im["height"]) for b in im["boxes"])


@pytest.mark.skipif(not (SAMPLES_DIR / "index.json").exists(), reason="samples not generated")
def test_samples_manifest_and_sizes():
    manifest = json.loads((SAMPLES_DIR / "index.json").read_text())
    assert [m["file"] for m in manifest] == [f"bridge_{k:02d}.jpg" for k in range(1, 9)]
    for m in manifest:
        assert m["synthetic"] is True and m["title"] and m["note"]
        assert all(b["cls"] in synth.CLASSES and _in_bounds(b["box"], m["width"], m["height"]) for b in m["boxes"])
        assert (SAMPLES_DIR / m["file"]).stat().st_size < synth.SAMPLE_LIMIT
    assert manifest[-1]["boxes"] == []  # clean concrete + formwork line: nothing real to find
