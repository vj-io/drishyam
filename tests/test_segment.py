from itertools import pairwise

import cv2
import numpy as np
import pytest

from drishyam import imaging
from drishyam.segment import CLASSES, ClassicalSegmenter, OnnxSegmenter, finish, mask_iou, segment_view

from .conftest import add_crack, add_rebar_spall, concrete, threaded_mismatches

SEG = ClassicalSegmenter()


def test_clean_concrete_is_quiet():
    seg = SEG.predict(concrete(512, 512, 11))
    assert sum(seg.areas.values()) < 0.01 * 512 * 512
    assert np.allclose(seg.probs.sum(axis=0), 1, atol=1e-4)


def test_crack_is_found_with_low_entropy_at_full_resolution():
    img = add_crack(concrete(512, 512, 12), (60, 200), n=120, width=3)
    seg = SEG.predict(img)
    assert seg.best == "crack" and seg.areas["crack"] > 300
    assert seg.entropy < 0.5


def test_hairline_crack_is_more_uncertain_when_downscaled():
    img = add_crack(concrete(1024, 1024, 13), (100, 500), n=260, width=1, step=(3, 0))
    fine = SEG.predict(img[256:768, 256:768])
    coarse = SEG.predict(cv2.pyrDown(img))
    assert fine.areas["crack"] > 0
    assert coarse.areas["crack"] * 4 < fine.areas["crack"] * 4 or coarse.entropy >= fine.entropy


def test_crack_that_fades_midway_stays_one_component():
    """Hysteresis: the faint middle of a crack is linked to its dark ends, not dropped."""
    img = concrete(512, 512, 19)
    pts = [(x, 256 + int(10 * np.sin(x / 40))) for x in range(60, 453)]
    for (x0, y0), (x1, y1) in pairwise(pts):
        v = int(145 - 90 * abs(x0 - 256) / 196)  # 55 at the ends, 145 on a ~170 surface mid-way
        cv2.line(img, (x0, y0), (x1, y1), (v, v, v + 3), 2, cv2.LINE_AA)
    n, _, stats, _ = cv2.connectedComponentsWithStats(SEG.predict(img).mask("crack"), connectivity=8)
    assert n > 1 and stats[1:, cv2.CC_STAT_WIDTH].max() > 0.9 * 392


def test_rust_in_spall_is_rebar_and_rust_alone_is_corrosion():
    img = add_rebar_spall(concrete(512, 512, 14), (256, 256), r=150)
    seg = SEG.predict(img)
    assert seg.areas["exposed_rebar"] > 200 and seg.areas["spalling"] > 1000
    stain = concrete(512, 512, 15)
    cv2.ellipse(stain, (256, 256), (90, 60), 0, 0, 360, (40, 80, 150), -1)
    seg = SEG.predict(stain)
    assert seg.best == "corrosion" and seg.areas["exposed_rebar"] < seg.areas["corrosion"] / 10


def test_a_border_crack_is_not_mirrored_outside_the_image():
    """A view wider than the image is filled by reflection, so a crack 15 px from the edge has a twin at -15 px.
    Only pixels that map inside the image may be labelled."""
    img = concrete(600, 800, 5)
    wobble = [(20 + round(5 * np.sin(y / 25)), y) for y in range(40, 561, 8)]  # a crack, not a ruler line
    cv2.polylines(img, [np.array(wobble, np.int32)], False, (55, 55, 58), 3, cv2.LINE_AA)
    crop, to_l0 = imaging.gaze(imaging.build_pyramid(img), 400, 300, 1, size=512)  # L0 window [-112, -212, 1024, 1024]
    valid = imaging.footprint(to_l0, (800, 600), 512)
    raw = SEG.predict(crop)
    assert np.count_nonzero(raw.mask("crack") & ~valid) > 1000  # the mirrored twin is there to be found
    seg = segment_view(SEG, crop, "accurate", valid)
    assert seg.areas["crack"] > 500 and not np.any(seg.labels[~valid])
    masked = raw.probs * valid
    masked[0] += ~valid
    full = finish(masked, 0.0, "accurate")  # the shortcut in `restrict` must equal a full re-finish
    assert np.array_equal(seg.labels, full.labels) and (seg.areas, seg.confidence, seg.entropy) == (
        full.areas, full.confidence, full.entropy)
    ys, xs = np.nonzero(seg.mask("crack"))
    x, y = imaging.map_points(to_l0, np.stack([xs, ys], 1)).T
    assert x.min() >= 0 and y.min() >= 0 and x.max() < 800 and y.max() < 600


def test_a_straight_groove_is_dropped_and_a_meandering_crack_is_kept():
    """To the evidence maps both are dark lines; only the crack meanders (drop_straight_lines). A coarse view
    smooths a crack's jitter away, so the residual is judged in L0 px: the same groove seen at L3 is kept."""
    everywhere = np.ones((512, 512), bool)
    groove = concrete(512, 512, 25)
    cv2.line(groove, (250, 10), (258, 500), (50, 52, 55), 3, cv2.LINE_AA)
    raw = SEG.predict(groove)
    assert raw.areas["crack"] > 1000
    seg = segment_view(SEG, groove, "accurate", everywhere)
    assert seg.areas["crack"] == 0 and len(seg.lines) == 1
    _box, length, width, residual = seg.lines[0]
    assert length > 400 and length > 20 * width and residual < 1.5
    assert segment_view(SEG, groove, "accurate", everywhere, scale=8).areas["crack"] == raw.areas["crack"]
    crack = add_crack(concrete(512, 512, 24), (60, 200), n=120, width=3)
    kept = segment_view(SEG, crack, "accurate", everywhere)
    assert kept.lines == [] and kept.areas["crack"] == SEG.predict(crack).areas["crack"] > 300


def test_fast_model_agrees_roughly_and_is_faster():
    img = add_crack(concrete(512, 512, 16), (60, 200), n=120, width=4)
    acc, fast = SEG.predict(img, "accurate"), SEG.predict(img, "fast")
    assert fast.model == "fast" and fast.areas["crack"] > 0
    assert mask_iou(cv2.dilate(acc.mask("crack"), None), cv2.dilate(fast.mask("crack"), None)) > 0.3


def test_finish_drops_specks_and_scores_entropy():
    probs = np.zeros((len(CLASSES), 64, 64), np.float32)
    probs[0] = 1.0
    probs[:, 5, 5] = 0
    probs[1, 5, 5] = 1.0  # one-pixel speck
    probs[0, 20:40, 20:40], probs[2, 20:40, 20:40] = 0.5, 0.5  # coin-flip spall block
    seg = finish(probs, 0.0, "t")
    assert seg.areas["crack"] == 0
    assert seg.entropy == pytest.approx(1.0, abs=0.01)


@pytest.fixture(scope="module")
def tiny_onnx(tmp_path_factory):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    w = numpy_helper.from_array(rng.normal(0, 0.5, (6, 3, 1, 1)).astype(np.float32), "w")
    node = helper.make_node("Conv", ["x", "w"], ["y"])
    graph = helper.make_graph([node], "seg", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, "h", "w"])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, "h", "w"])], [w])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path_factory.mktemp("m") / "tiny.onnx"
    onnx.save(model, path)
    return str(path)


def test_micro_times_nanodet_and_the_onnx_named_by_the_env(tiny_onnx, monkeypatch):
    from bench import micro

    monkeypatch.delenv("DRISHYAM_ONNX_FP16", raising=False)
    monkeypatch.delenv("DRISHYAM_ONNX", raising=False)
    assert set(micro.dnn_ops()) == {"nanodet"}  # nothing to time but the privacy detector without a model
    monkeypatch.setenv("DRISHYAM_ONNX", tiny_onnx)
    table = micro.dnn_ops()
    assert set(table) == {"nanodet", "onnx:fp32@256", "onnx:fp32@512"}
    for name, (s, forward) in table.items():
        if name.startswith("onnx:"):
            assert np.asarray(forward()).shape == (1, 6, s, s)


@pytest.mark.parametrize("shape", [(512, 512), (320, 480), (256, 256)])
def test_onnx_engine_new_dynamic_hw(tiny_onnx, shape):
    seg = OnnxSegmenter(tiny_onnx).predict(concrete(*shape, 17))
    assert seg.probs.shape == (6, *shape)
    assert np.allclose(seg.probs.sum(axis=0), 1, atol=1e-4)
    assert seg.model == "accurate"


def test_onnx_class_map_merges_channels(tiny_onnx):
    seg = OnnxSegmenter(tiny_onnx, class_map=[0, 0, 1, 1, 2, 2]).predict(concrete(128, 128, 18))
    assert np.allclose(seg.probs[3:], 0) and np.allclose(seg.probs.sum(axis=0), 1, atol=1e-4)


def test_eval_cache_identity_tells_same_named_models_and_the_fp16_net_apart(tiny_onnx, tmp_path):
    """evals.harness --reuse compares this: every install is models/drishyam_dacl10k.onnx, so the name is not enough."""
    import onnx
    from onnx import numpy_helper

    from evals.harness import segmenter_id

    m = onnx.load(tiny_onnx)  # the same graph and file name, other weights: a newer training run
    m.graph.initializer[0].CopyFrom(numpy_helper.from_array(-numpy_helper.to_array(m.graph.initializer[0]), "w"))
    retrained = tmp_path / "tiny.onnx"
    onnx.save(m, retrained)
    segs = [OnnxSegmenter(tiny_onnx), OnnxSegmenter(tiny_onnx, tiny_onnx), OnnxSegmenter(str(retrained))]
    assert {s.name for s in segs} == {"onnx:tiny.onnx"}  # summary.json's readable label
    assert len({segmenter_id(s) for s in segs}) == 3
    assert segmenter_id(SEG) == SEG.name


@pytest.mark.parametrize("engine", ["classical", "onnx"])
def test_a_shared_segmenter_gives_each_thread_its_own_result(engine, request):
    """default_segmenter() is one instance per process and the web app runs concurrent jobs' steps in
    a threadpool, so no call may see another thread's crop (CLAHE buffers, a net's set input)."""
    seg = SEG if engine == "classical" else OnnxSegmenter(request.getfixturevalue("tiny_onnx"), input_size=128)
    crops = [add_crack(concrete(256, 256, 30), (10, 60), n=80), concrete(256, 256, 31)]
    assert threaded_mismatches(lambda c: seg.predict(c).probs.tobytes(), crops) == []
