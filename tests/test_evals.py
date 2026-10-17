"""Evaluation harness, CODEBRIM loader and parity tool: matching rule, tiling accounting, an end-to-end tiny run."""
import json
import zipfile

import cv2
import numpy as np
import pytest

from bench import parity
from drishyam.segment import CLASSES, finish
from evals import codebrim, failures, harness, synth


class BlankSegmenter:
    """Predicts 'all background' instantly, so pixel accounting is tested without paying for segmentation."""

    def predict(self, crop, model="accurate"):
        h, w = crop.shape[:2]
        probs = np.zeros((len(CLASSES), h, w), np.float32)
        probs[0] = 1
        return finish(probs, 0.0, model)


# ---------------------------------------------------------------- matching
def test_match_by_iou_or_centre_and_class():
    gt = {"cls": "crack", "box": [100, 100, 100, 100]}
    p = lambda cls, box: [{"cls": cls, "box": box}]
    assert harness.found(gt, p("crack", [100, 100, 100, 100]))  # perfect overlap
    assert harness.found(gt, p("crack", [150, 100, 100, 100]))  # IoU 1/3 >= 0.3
    assert not harness.found(gt, p("crack", [170, 100, 100, 100]))  # IoU 0.18 and centre (150,150) outside
    assert harness.found(gt, p("crack", [0, 0, 400, 400]))  # IoU tiny, but it contains the GT centre
    assert not harness.found(gt, [])  # nothing predicted
    assert not harness.found(gt, p("spalling", [100, 100, 100, 100]))  # right place, wrong class
    assert not harness.found(gt, p("crack", [0, 0, 400, 400]), centre=False)  # strict variant ignores containment
    assert harness.hits([140, 140, 20, 20], [100, 100, 100, 100]) and not harness.hits([0, 0, 50, 50], [100, 100, 100, 100])


def test_critical_false_accepts():
    rebar = {"cls": "exposed_rebar", "box": [0, 0, 100, 100]}
    small_spall = {"cls": "spalling", "box": [300, 0, 100, 100]}  # box 10,000 px^2: below the critical area
    big_spall = {"cls": "spalling", "box": [500, 0, 300, 300]}
    gts = [rebar, small_spall, big_spall]
    preds = [{"cls": "exposed_rebar", "box": [0, 0, 100, 100], "area": 5000},  # matches rebar GT
             {"cls": "exposed_rebar", "box": [900, 900, 50, 50], "area": 100},  # nothing there
             {"cls": "spalling", "box": [300, 0, 100, 100], "area": 60_000},  # critical by area, GT spall is not
             {"cls": "spalling", "box": [500, 0, 300, 300], "area": 60_000},  # matches the big spall
             {"cls": "crack", "box": [0, 0, 10, 10], "area": 10}]  # never critical
    assert harness.critical_counts(preds, gts) == (4, 2)


# ---------------------------------------------------------------- tiling and downscale accounting
@pytest.mark.parametrize("h,w", [(1536, 2048), (768, 1024), (500, 600), (300, 400), (3000, 4000)])
def test_tiles_cover_the_image_with_the_right_overlap(h, w):
    boxes = harness.slice_boxes(h, w)
    cover = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        assert x1 - x0 == min(512, w) and y1 - y0 == min(512, h)  # edge tiles are pulled back, not shrunk
        cover[y0:y1, x0:x1] = True
    assert cover.all()
    xs = sorted({b[0] for b in boxes})
    if w >= 922:  # at least three columns, so the second tile is a regular one and not the pulled-back edge tile
        assert xs[1] - xs[0] == 512 - int(0.2 * 512)  # stride = tile x (1 - overlap): a 20 % overlap
    assert len(harness.slice_boxes(1536, 2048)) == 20  # 5 columns x 4 rows


def test_tiling_pixel_accounting():
    img = np.zeros((768, 1024, 3), np.uint8)
    l0 = harness.run_tiling(img, BlankSegmenter(), 0)
    tiles = harness.slice_boxes(768, 1024)
    assert len(tiles) == 6
    assert l0["pixels_frac"] == pytest.approx(len(tiles) * 512 * 512 / (768 * 1024))  # overlap counted
    l1 = harness.run_tiling(img, BlankSegmenter(), 1)  # level 1 is 512x384: one 512x384 tile
    assert l1["pixels_frac"] == pytest.approx(512 * 384 / (768 * 1024))
    assert l1["preds"] == [] and l0["wall_s"] >= 0


def test_downscale_pixel_accounting():
    big = harness.run_downscale(np.zeros((1500, 2000, 3), np.uint8), BlankSegmenter())
    assert big["pixels_frac"] == pytest.approx(1024 * 768 / (1500 * 2000), rel=1e-3)  # 1024 px on the long side
    small = harness.run_downscale(np.zeros((300, 400, 3), np.uint8), BlankSegmenter())
    assert small["pixels_frac"] == 1.0  # never upscaled


def test_predictions_map_back_to_l0():
    h, w = 128, 128
    probs = np.zeros((len(CLASSES), h, w), np.float32)
    probs[0] = 1
    probs[0, 20:40, 30:70], probs[1, 20:40, 30:70] = 0, 1  # a crack blob at crop (30, 20) 40x20
    seg = finish(probs, 0.0, "accurate")
    to_l0 = np.array([[2.0, 0, 2 * 100], [0, 2.0, 2 * 50], [0, 0, 1]])  # tile at (100, 50) of pyramid level 1
    (comp,) = harness.preds_from_seg(seg, to_l0)
    assert comp["cls"] == "crack" and comp["box"] == [2 * 130, 2 * 70, 80, 40] and comp["area"] == 4 * 800
    assert harness.union_from_seg(seg, to_l0)[0]["box"] == comp["box"]


def test_drishyam_components_reproduce_finding_boxes(scene):
    """The rebuilt per-component boxes of every look in a finding's trail must sit inside its own (union) box."""
    rec = harness.RecordingSegmenter()
    job = harness.Job(scene, harness.JobConfig(pixel_cap=0.3, thumbs=False), segmenter=rec)
    list(job.run())
    comps = harness.finding_components(job, rec.log)
    assert job.findings and comps
    for c in comps:
        f = next(f for f in job.findings if f.cls == c["cls"] and c["step"] in f.steps)
        fx, fy, fw, fh = f.box
        assert c["box"][0] >= fx - 2 and c["box"][1] >= fy - 2
        assert c["box"][0] + c["box"][2] <= fx + fw + 2 and c["box"][1] + c["box"][3] <= fy + fh + 2


def test_chaos_drills_recover(scene):
    # Real clock, real stop rule: the job may halt on the step right after the p95 spike; the governor's
    # model switch is then recorded on that halt record (`governor`), which the drill accepts.
    out = harness.run_chaos(scene, "planner_timeout")
    assert out["fired"] and out["ok"] and out["detail"]["reason"] == "timeout"
    spike = harness.run_chaos(scene, "p95_spike")
    assert spike["fired"] and spike["ok"]


def test_governor_activity_counts_overrides_noted_on_halt_records():
    trace = [{"action": "gaze", "cause": "survey"}, {"action": "gaze", "cause": "p95>SLA"},
             {"action": "halt", "cause": "stop-rule", "governor": "warp-spike"}]
    res = [{"width": 100, "height": 100, "configs": {"drishyam@0.3": {"trace": trace, "pixels_frac": 0.2, "halt_cause": "stop-rule"}}}]
    gov = harness.agent_metrics(res, [], (0.3,), 1)["governor_activity_in_eval"]
    assert gov == {"p95>SLA": 1, "warp-spike": 1, "usd≥80%": 0}


# ---------------------------------------------------------------- end to end
def test_three_image_run_writes_valid_summary_and_gallery(tmp_path):
    data = tmp_path / "data"
    synth.build_dataset(data, n=3, seed=7, width=640, height=480)
    pub, raw = tmp_path / "pub", tmp_path / "raw"
    summary = harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), log=lambda *_: None)
    on_disk = json.loads((pub / "summary.json").read_text())
    assert on_disk == json.loads(json.dumps(summary))
    assert list(on_disk) == ["dataset", "n_images", "segmenter", "policies", "per_class", "agent_metrics", "micro", "provenance", "headline"]
    assert on_disk["n_images"] == 3 and on_disk["dataset"].startswith("synthetic-3")
    pol = on_disk["policies"]
    assert [len(pol[k]) for k in ("downscale", "tiling", "drishyam")] == [1, 3, 6]
    for points in pol.values():
        for p in points:
            assert set(p) == {"pixels_frac", "recall", "usd"} and 0 <= p["recall"] <= 1 and p["pixels_frac"] > 0 and p["usd"] >= 0
    assert {"drishyam@0.3", "tiling@1.0", "downscale"} <= set(on_disk["per_class"])
    assert set(on_disk["per_class"]["downscale"]) == set(synth.CLASSES)
    assert set(on_disk["provenance"]) == {"cv2", "build_info_sha256", "machine", "cpu", "python", "timestamp", "build"}
    am = on_disk["agent_metrics"]
    assert am["steps_with_cause_pct"] == 100.0 and set(am["box_convention"]["recall"]) == {"components", "union", "as_specified"}
    head = on_disk["headline"]
    assert isinstance(head, str) and head.count(". ") == 1  # two sentences
    for baseline in ("20%-overlap 512-px tiling", "1024-px downscale", "L1 tiling", "L2 tiling", "IoU >= 0.3"):
        assert baseline in head, baseline
    gallery = json.loads((pub / "failures.json").read_text())
    for g in gallery:
        assert set(g) == {"image", "category", "what_happened", "rule_that_would_catch_it"}
        assert (pub / g["image"]).exists()  # paths in failures.json are relative to public/results/ (the page prepends results/)
    assert (raw / "eval_raw.json").exists() and (raw / "eval_per_image.json").exists()
    again = harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), reuse=True, log=lambda *_: None)
    assert again["policies"]["drishyam"] == summary["policies"]["drishyam"]  # same predictions, re-scored
    cache = raw / "eval_per_image.json"
    saved = json.loads(cache.read_text())
    assert saved["segmenter"] == summary["segmenter"] == "classical-v1"
    cache.write_text(json.dumps({**saved, "segmenter": "onnx:m.onnx"}))  # an ONNX run, re-scored without DRISHYAM_ONNX
    with pytest.raises(SystemExit, match="onnx:m.onnx predictions"):
        harness.run(str(data), n=3, out_dir=pub, results_dir=raw, workers=1, chaos=(0, 0, 0), reuse=True, log=lambda *_: None)


def test_gallery_picks_a_real_false_alarm(tmp_path):
    path = tmp_path / "s0001.jpg"
    cv2.imwrite(str(path), np.full((600, 800, 3), 150, np.uint8))
    rec = {"id": "s0001", "path": str(path), "boxes": [], "distractors": [{"kind": "graffiti", "box": [100, 100, 200, 200]}]}
    pred = {"cls": "corrosion", "box": [120, 120, 100, 100], "area": 5000, "conf": 0.8, "step": 1, "level": 1, "region": "r1"}
    quiet = {"preds": [], "trace": []}
    res = {"configs": {"drishyam@0.3": {"preds": [pred], "trace": []}, "drishyam@1.0": quiet, "tiling@L0": quiet}}
    (row,) = failures.build_gallery([rec], [res], tmp_path / "pub", log=lambda *_: None)
    assert row["category"] == "Graffiti read as damage" and "corrosion" in row["what_happened"]
    assert (tmp_path / "pub" / "failures" / "f01.jpg").exists() and row["image"] == "failures/f01.jpg"


def test_render_case_is_narrow_enough(tmp_path):
    img = np.full((1536, 2048, 3), 150, np.uint8)
    case = {"marks": [{"kind": "gt", "label": "GT crack", "box": [100, 100, 900, 700]},
                      {"kind": "pred", "label": "crack 0.9", "box": [150, 120, 200, 200]}]}
    trace = [{"action": "gaze", "x": 400, "y": 300, "level": 1, "window": [0, 0, 1024, 1024]},
             {"action": "halt", "x": None, "y": None, "level": None, "window": None}]
    out = tmp_path / "f01.jpg"
    failures.render_case(img, case, trace, "f01  Test  |  s0001  |  drishyam@0.3", out)
    shown = cv2.imread(str(out))
    assert shown is not None and shown.shape[1] <= 900


# ---------------------------------------------------------------- CODEBRIM
XML = """<annotation>
  <filename>image_0000001.jpg</filename>
  <size><width>6000</width><height>4000</height><depth>3</depth></size>
  <object><name>defect</name><bndbox><xmin>100</xmin><ymin>200</ymin><xmax>400</xmax><ymax>260</ymax></bndbox>
    <Defect><Background>0</Background><Crack>1</Crack><Spallation>0</Spallation><Efflorescence>0</Efflorescence>
    <ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
  <object><name>defect</name><bndbox><xmin>1000</xmin><ymin>1000</ymin><xmax>1500</xmax><ymax>1800</ymax></bndbox>
    <Defect><Background>0</Background><Crack>0</Crack><Spallation>1</Spallation><Efflorescence>0</Efflorescence>
    <ExposedBars>1</ExposedBars><CorrosionStain>1</CorrosionStain></Defect></object>
  <object><name>defect</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
    <Defect><Background>1</Background><Crack>0</Crack><Spallation>0</Spallation><Efflorescence>0</Efflorescence>
    <ExposedBars>0</ExposedBars><CorrosionStain>0</CorrosionStain></Defect></object>
</annotation>"""  # the real release's layout: the flags sit in <Defect>


def test_codebrim_xml_parsing_maps_flags_to_our_classes():
    a = codebrim.parse_xml(XML)
    assert (a["file"], a["width"], a["height"]) == ("image_0000001.jpg", 6000, 4000)
    assert a["boxes"] == [{"cls": "crack", "box": [100, 200, 300, 60]},
                          {"cls": "spalling", "box": [1000, 1000, 500, 800]},  # one box, three flags: three GT entries
                          {"cls": "exposed_rebar", "box": [1000, 1000, 500, 800]},
                          {"cls": "corrosion", "box": [1000, 1000, 500, 800]}]  # the background-only box is dropped


def test_codebrim_load_and_seeded_selection(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    rng = np.random.default_rng(0)
    for i in range(30):
        flags = {c: int(rng.random() < 0.3) for c in ("Crack", "Spallation", "Efflorescence", "ExposedBars", "CorrosionStain")}
        body = "".join(f"<{k}>{v}</{k}>" for k, v in flags.items())
        (tmp_path / "images" / f"image_{i:07d}.jpg").write_bytes(b"x")
        (tmp_path / "annotations" / f"image_{i:07d}.xml").write_text(
            f"<annotation><filename>image_{i:07d}.jpg</filename><size><width>10</width><height>10</height></size>"
            f"<object><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox>"
            f"<Defect>{body}</Defect></object></annotation>")
    fork = tmp_path / "__MACOSX" / "annotations"  # the zip ships binary macOS resource forks named like the XML
    fork.mkdir(parents=True)
    (fork / "._image_0000000.xml").write_bytes(b"\x00\x05\x16\x07\xff\xfe")
    records = codebrim.load(tmp_path)
    assert len(records) == 30 and records[0]["id"] == "image_0000000"
    subset, short = codebrim.select_eval(records, n=12, seed=7, min_per_class=3)
    assert len(subset) == 12 and not short
    for c in codebrim.CLASSES:
        assert sum(b["cls"] == c for r in subset for b in r["boxes"]) >= 3
    assert [r["id"] for r in subset] == [r["id"] for r in codebrim.select_eval(records, n=12, seed=7, min_per_class=3)[0]]
    out = tmp_path / "eval.txt"
    codebrim.write_list(subset, out)
    assert [r["id"] for r in codebrim.load_eval(tmp_path, out)] == [r["id"] for r in subset]



def test_codebrim_unpack_survives_wrapped_offsets(tmp_path):
    """The release zip's offsets wrap at 2**32; every entry must still come out intact, forks skipped, paths flattened."""
    zp = tmp_path / "release.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("original_dataset/images/image_0000001.jpg", b"\xff\xd8jpeg")
        z.writestr("original_dataset/annotations/image_0000001.xml", XML)
        z.writestr("__MACOSX/original_dataset/annotations/._image_0000001.xml", b"\x00\x05\x16\x07")
        z.writestr("../../escape.xml", "<annotation/>")
    with zipfile.ZipFile(zp) as z:
        for i, info in enumerate(z.infolist()):  # how zipfile sees the real file: some +2**32, some as stored
            info.header_offset += 2**32 if i % 2 == 0 else 0
        got = dict(codebrim._members(z, zp.stat().st_size))
    assert got == {"image_0000001.jpg": b"\xff\xd8jpeg", "image_0000001.xml": XML.encode(), "escape.xml": b"<annotation/>"}
    assert codebrim.unpack(zp, tmp_path / "out") == 3
    assert sorted(p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*.*")) == \
        ["annotations/escape.xml", "annotations/image_0000001.xml", "images/image_0000001.jpg"]

def test_harness_refuses_to_publish_codebrim_crops(tmp_path):
    """CODEBRIM's licence bars redistributing its images, so the gallery crops must never land under public/."""
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "image_0000001.jpg").write_bytes(b"x")
    (tmp_path / "image_0000001.xml").write_text(XML)
    (tmp_path / "eval200.txt").write_text("image_0000001.jpg\n")
    with pytest.raises(SystemExit, match="refusing"):
        harness.run(tmp_path, n=1, out_dir=tmp_path / "public" / "results", results_dir=tmp_path / "raw", workers=1)

# ---------------------------------------------------------------- parity
def test_parity_compare_saved_files(tmp_path):
    a = np.zeros((2, 8, 8), np.uint8)
    a[:, :4] = 1  # half crack
    b = a.copy()
    b[1, :2] = 0  # 16 of the 128 crack pixels become background in the second crop
    crops = np.zeros((2, 8, 8, 3), np.uint8)
    for name, labels in (("a.npz", a), ("b.npz", b)):
        np.savez(tmp_path / name, crops=crops, labels=labels, classes=np.array(CLASSES), meta=json.dumps(
            {"cv2": "x", "build": "y", "cpu": "z"}))
    res = parity.compare_files(str(tmp_path / "a.npz"), str(tmp_path / "b.npz"))
    assert res["crops"] == 2 and res["inputs_identical"] and res["identical_crops"] == 1
    assert res["pixel_agreement"] == pytest.approx(1 - 16 / 128)
    assert res["worst_crop_agreement"] == pytest.approx(1 - 16 / 64)
    assert res["class_iou"]["crack"] == pytest.approx(48 / 64) and res["class_iou"]["spalling"] is None
    with pytest.raises(ValueError):
        parity.compare(a, b[:1])
