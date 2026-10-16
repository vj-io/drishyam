import cv2
import numpy as np
import pytest

from drishyam.segment import CLASSES, ClassicalSegmenter, OnnxSegmenter, finish, mask_iou

from .conftest import add_crack, add_rebar_spall, concrete

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


def test_rust_in_spall_is_rebar_and_rust_alone_is_corrosion():
    img = add_rebar_spall(concrete(512, 512, 14), (256, 256), r=150)
    seg = SEG.predict(img)
    assert seg.areas["exposed_rebar"] > 200 and seg.areas["spalling"] > 1000
    stain = concrete(512, 512, 15)
    cv2.ellipse(stain, (256, 256), (90, 60), 0, 0, 360, (40, 80, 150), -1)
    seg = SEG.predict(stain)
    assert seg.best == "corrosion" and seg.areas["exposed_rebar"] < seg.areas["corrosion"] / 10


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


@pytest.mark.parametrize("shape", [(512, 512), (320, 480), (256, 256)])
def test_onnx_engine_new_dynamic_hw(tiny_onnx, shape):
    seg = OnnxSegmenter(tiny_onnx).predict(concrete(*shape, 17))
    assert seg.probs.shape == (6, *shape)
    assert np.allclose(seg.probs.sum(axis=0), 1, atol=1e-4)
    assert seg.model == "accurate"


def test_onnx_class_map_merges_channels(tiny_onnx):
    seg = OnnxSegmenter(tiny_onnx, class_map=[0, 0, 1, 1, 2, 2]).predict(concrete(128, 128, 18))
    assert np.allclose(seg.probs[3:], 0) and np.allclose(seg.probs.sum(axis=0), 1, atol=1e-4)
