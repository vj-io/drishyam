import hashlib
import json
import platform
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
        if platform.machine() in ("x86_64", "AMD64"):  # recorded on x86: the aarch64 HAL (carotene, KleidiCV) rounds some pixel ops differently
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
    assert ann["generator_version"] == synth.GENERATOR_VERSION and ann["clean"] is False
    assert [i["id"] for i in ann["images"]] == ["s0001", "s0002", "s0003"]
    for im in ann["images"]:
        assert cv2.imread(str(tmp_path / im["file"])).shape == (im["height"], im["width"], 3)
        assert all(_in_bounds(b["box"], im["width"], im["height"]) for b in im["boxes"])


def test_clean_set_has_distractors_and_no_damage(tmp_path):
    synth.build_dataset(tmp_path, n=4, seed=11, width=320, height=240, clean=True)
    ann = json.loads((tmp_path / "annotations.json").read_text())
    assert ann["clean"] is True and ann["generator_version"] == synth.GENERATOR_VERSION and ann["dataset"] == "synthetic-clean-5"
    assert [i["id"] for i in ann["images"]] == ["s0001", "s0002", "s0003", "s0004", "bridge_08"]
    for im in ann["images"]:
        assert im["boxes"] == [] and 1 <= len(im["distractors"]) <= 2
        assert {d["kind"] for d in im["distractors"]} <= set(synth.CLEAN_DISTRACTORS)
    assert ann["images"][-1]["width"] == 3000
    if (SAMPLES_DIR / "bridge_08.jpg").exists():  # the demo's clean sample, byte for byte
        assert (tmp_path / "images" / "bridge_08.jpg").read_bytes() == (SAMPLES_DIR / "bridge_08.jpg").read_bytes()


@pytest.mark.skipif(not (SAMPLES_DIR / "index.json").exists(), reason="samples not generated")
def test_samples_manifest_and_sizes():
    manifest = json.loads((SAMPLES_DIR / "index.json").read_text())
    made = [m for m in manifest if m["synthetic"] is True]
    assert [m["file"] for m in made] == [f"bridge_{k:02d}.jpg" for k in range(1, 9)]
    for m in made:
        assert m["title"] and m["note"]
        assert all(b["cls"] in synth.CLASSES and _in_bounds(b["box"], m["width"], m["height"]) for b in m["boxes"])
    assert made[-1]["boxes"] == []  # clean concrete + formwork line: nothing real to find
    # the team's own wall photos: labelled as such, no ground truth, each with the recorded v1 run on that very file
    real = [m for m in manifest if m["synthetic"] is False]
    assert [m["file"] for m in real] == ["real_01.jpg", "real_02.jpg"] and len(made) + len(real) == len(manifest)
    for m in real:
        assert m["title"] == "REAL PHOTO (ours), painted plaster wall, no ground truth" and "not ground truth" in m["note"]
        assert "boxes" not in m and "distractors" not in m
        ev = [json.loads(ln) for ln in (SAMPLES_DIR / m["trace"]).read_text().splitlines()]
        assert cv2.imread(str(SAMPLES_DIR / m["file"])).shape[1::-1] == (m["width"], m["height"]) == (ev[0]["width"], ev[0]["height"])
        assert ev[0]["cost_model"]["micro_arm"].endswith("-dacl10k-retime") and ev[-1]["work_order"]["profile"] == "onnx:drishyam_dacl10k.onnx"
        assert "token" not in ev[0] and "kill_token" not in ev[0] and "token" not in ev[-1]
    for m in manifest:
        assert (SAMPLES_DIR / m["file"]).stat().st_size < synth.SAMPLE_LIMIT and (SAMPLES_DIR / m["thumb"]).exists()


def test_sample_jpegs_carry_no_exif():
    """Every JPEG under public/samples ships with no metadata but its JFIF header: EXIF and XMP (where a camera puts GPS)
    are APP1 segments, IPTC is APP13. The team's photos are copied from private-assets/real-photos/clean/ only."""
    jpegs = sorted(p for p in SAMPLES_DIR.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    assert len(jpegs) >= 20
    for p in jpegs:
        b = p.read_bytes()
        assert b[:2] == b"\xff\xd8", p
        i, apps = 2, []
        while b[i] == 0xFF and b[i + 1] != 0xDA:  # every marker segment up to the start of scan
            if 0xE0 <= b[i + 1] <= 0xEF:
                apps.append((hex(b[i + 1]), b[i + 4:i + 9]))
            i += 2 + int.from_bytes(b[i + 2:i + 4], "big")
        assert apps == [("0xe0", b"JFIF\0")], (p.name, apps)
