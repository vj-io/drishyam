"""drishyam.segment's OpenCV flags leave every output the same to the byte: the agent's traces on bundled photos, and
the units they replace on hard cases (ties, specks, a ragged footprint); OnnxSegmenter's engine option; and
bench/opencv5_features.py on a tiny model; two OpenCV 5.0.0 DNN quirks; and docs/bench.md's "OpenCV 5 features in
the pipeline" is the committed JSON (results/local-macos-x86/opencv5_features.json and opencv_builds.json), the
shipped flag values included. `python -m tests.test_opencv5_features` prints the section's rows."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from bench import opencv5_features as feat
from drishyam import segment
from drishyam.segment import ClassicalSegmenter, OnnxSegmenter
from evals import synth
from tests.conftest import concrete


def test_flags_leave_the_agents_traces_byte_identical():
    """All flags off (the NumPy forms) against all on, on two demo samples, a real photo and a synth_dev image: the
    whole event stream (gaze path, causes, findings, thumbnails, work order) is the same string."""
    imgs = {k: v for k, v in feat.photos(1).items() if k in ("bridge_01", "bridge_03", "real_02", "synth_dev/s0001")}
    assert len(imgs) == 4
    seg, looks = ClassicalSegmenter(), []
    for name, img in imgs.items():
        with feat.flags():
            off = feat.trace(img, seg, looks)
        with feat.flags(feat.FLAGS):
            assert feat.trace(img, seg) == off, name
    assert any(not v.all() for _, _, v, _ in looks)  # some look ran past the image: restrict's masked copy ran


def test_finish_and_restrict_match_numpy_on_ties_specks_and_a_ragged_footprint():
    rng = np.random.default_rng(3)
    probs = rng.integers(1, 4, (len(segment.CLASSES), 97, 131)).astype(np.float32)  # many exact ties
    probs /= probs.sum(0, keepdims=True)
    valid = rng.random((97, 131)) > 0.3
    got = []
    for on in ((), feat.FLAGS):
        with feat.flags(on):
            s = segment.finish(probs, 0.0, "accurate")
            got.append((s, segment.restrict(s, valid)))
    for a, b in zip(*got, strict=True):
        assert a.labels.tobytes() == b.labels.tobytes() and a.probs.tobytes() == b.probs.tobytes()
        assert (a.areas, a.confidence, a.entropy) == (b.areas, b.confidence, b.entropy)
    assert len(set(got[0][0].areas.values())) > 1 and got[0][1].areas != got[0][0].areas  # specks and the mask bit


def test_onnx_engine_classic_runs_a_dynamic_model_through_a_static_copy(tiny_onnx):
    crop = concrete(320, 480, 5)  # not square: letterboxed into the 512 (and 256) input
    new, classic = (OnnxSegmenter(tiny_onnx, engine=e) for e in ("new", "classic"))
    assert classic.engine == "classic" and classic.nets["accurate"] is not classic.nets["fp16"]
    # ENGINE_NEW adds layout transforms ('trlayout.0'); 5.0.0's readNetFromONNX(buffer, ENGINE_CLASSIC) returned one
    assert all(not any(n.startswith("trlayout") for n in net.getLayerNames()) for net in classic.nets.values())
    assert any(n.startswith("trlayout") for n in new.nets["accurate"].getLayerNames())
    for model in ("accurate", "fast"):
        a, b = new.predict(crop, model), classic.predict(crop, model)
        assert np.abs(a.probs - b.probs).max() < 1e-5 and (a.labels == b.labels).mean() > 0.999
    with pytest.raises(ValueError, match="'new' or 'classic'"):
        OnnxSegmenter(tiny_onnx, engine="ort")


def test_bench_parts_on_a_tiny_model(tiny_onnx):
    img = {"synth": synth.generate(9101, 2048, 1536)[0]}
    seg = ClassicalSegmenter()
    ident, looks = feat.identity(seg, img)
    assert ident["identical"] == dict.fromkeys(["null", *feat.FLAGS, "all"], 1) and ident["looks"] == len(looks) > 0
    t = feat.timing(seg, looks, 2, 3, [1])
    assert set(t) == set(feat.UNITS) and t["all"]["segment_view"]["threads1"]["on"]["ratio"] > 0
    assert len(t["all"]["segment_view"]["threads1"]["null"]["ci"]) == 2
    prof = feat.profile(tiny_onnx, (64,), [1], 2)[64]["threads1"]
    assert "TransformLayout" in prof["new"]["by_kind"] and "TransformLayout" not in prof["classic"]["by_kind"]
    for e in ("new", "classic"):
        assert sum(k["layers"] for k in prof[e]["by_kind"].values()) == prof[e]["layers"] >= 1
    eng = feat.engine([tiny_onnx], img, 1, 2, [1])
    assert eng["photos"] == 1 and eng["max_dprob"] < 1e-5 and "threads1" in eng


def test_two_opencv_5_0_0_dnn_quirks(tiny_onnx):
    """readNetFromONNX(buffer, engine=ENGINE_CLASSIC) returns an ENGINE_NEW net (so segment._classic reads a file), and
    enableFusion(False) leaves an ENGINE_NEW net empty. A wheel that changes either fails here."""
    layout = lambda net: any(n.startswith("trlayout") for n in net.getLayerNames())
    data = np.frombuffer(Path(tiny_onnx).read_bytes(), np.uint8)
    assert layout(cv2.dnn.readNetFromONNX(data, engine=cv2.dnn.ENGINE_CLASSIC))
    assert not layout(cv2.dnn.readNetFromONNX(tiny_onnx, engine=cv2.dnn.ENGINE_CLASSIC))
    net = cv2.dnn.readNetFromONNX(tiny_onnx, engine=cv2.dnn.ENGINE_NEW)
    net.enableFusion(False)
    net.setInput(np.zeros((1, 3, 16, 16), np.float32))
    with pytest.raises(cv2.error, match="empty"):
        net.forward()


REPO = Path(__file__).resolve().parents[1]
MAC = "local-macos-x86"
SEGS = {"classical": "classical-v1", "v1": "v1 on `ENGINE_NEW`"}
VARIANTS = ("null", *feat.FLAGS, "all")


def load(name: str, arm: str = MAC) -> dict:
    return json.loads((REPO / "results" / arm / name).read_text())


def section() -> str:
    text = (REPO / "docs" / "bench.md").read_text().split("## OpenCV 5 features in the pipeline\n")[1].split("\n## ")[0]
    return " ".join(text.split())


def iv(r: dict) -> str:
    return f"x{r['ratio']:.2f} ({r['ci'][0]:.2f}-{r['ci'][1]:.2f})"


def adopted(doc: dict, flag: str) -> bool:
    """The rule the shipped flags follow: the agent's traces identical on every photo with every segmenter, and on
    every unit of the flag, at every thread count, an on/off interval wholly below 1."""
    same = all(i["identical"][flag] == i["photos"] for i in doc["identity"].values())
    rows = [doc["timing"][s][flag][u] for s in doc["timing"] for u in feat.UNITS[flag]]
    return same and all(r[k]["on"]["ci"][1] < 1 for r in rows for k in r if k.startswith("threads"))


def identity_rows():
    for s, i in load("opencv5_features.json")["identity"].items():
        yield (f"| {SEGS[s]} | {i['photos']} | {i['looks']} | {i['events']} | "
               + " | ".join(f"{i['identical'][v]} of {i['photos']}" for v in VARIANTS) + " |")


def timing_rows():
    doc = load("opencv5_features.json")
    for flag, units in feat.UNITS.items():
        for s, t in doc["timing"].items():
            for u in units:
                r = t[flag][u]
                for k in (k for k in r if k.startswith("threads")):
                    x = r[k]
                    yield (f"| {'all three' if flag == 'all' else flag} | {SEGS[s]} | `{u}` on {r['inputs']} looks | "
                           f"{k.removeprefix('threads')} | {x['off_p50_ms']:.1f} | {x['on_p50_ms']:.1f} | {iv(x['on'])} | "
                           f"{iv(x['null'])} |")


def profile_rows(size: int = 512, top: int = 8):
    """The kinds of layer that take the most of the new engine's time at `size`, on 1 thread and on all."""
    doc = load("opencv5_features.json")
    p = doc["profile"][str(size)]
    one, many = p["threads1"], p[f"threads{doc['config']['all_threads']}"]
    cell = lambda row, e, k: f"{row[e]['by_kind'][k]['ms']:.1f}" if k in row[e]["by_kind"] else ""
    count = lambda e, k: str(one[e]["by_kind"][k]["layers"]) if k in one[e]["by_kind"] else "0"
    for k in list(one["new"]["by_kind"])[:top]:
        yield (f"| {k} | {count('new', k)} | {count('classic', k)} | {cell(one, 'new', k)} | {cell(one, 'classic', k)} | "
               f"{cell(many, 'new', k)} | {cell(many, 'classic', k)} |")
    yield (f"| every layer (the profiler's sum) | {one['new']['layers']} | {one['classic']['layers']} | "
           f"{one['new']['profiled_ms']:.1f} | {one['classic']['profiled_ms']:.1f} | {many['new']['profiled_ms']:.1f} | "
           f"{many['classic']['profiled_ms']:.1f} |")
    yield (f"| a forward, wall time (p50) | | | {one['new']['wall_p50_ms']:.1f} | {one['classic']['wall_p50_ms']:.1f} | "
           f"{many['new']['wall_p50_ms']:.1f} | {many['classic']['wall_p50_ms']:.1f} |")


def builds_rows():
    v = load("opencv_builds.json")["variants"]
    others = [k for k in v if k not in ("5.0", "5.0 null")]
    for s, ref in v["5.0"]["stages"].items():
        cells = [f"{iv(v[o]['stages'][s]['vs_5.0'])}, {'same' if v[o]['stages'][s]['same_output'] else 'differs'}"
                 for o in others]
        yield f"| `{s}` | {ref['p50_ms']:.1f} | " + " | ".join(cells) + f" | {iv(v['5.0 null']['stages'][s]['vs_5.0'])} |"


def api_rows():
    v = load("opencv_builds.json")["variants"]
    others = [k for k in v if k not in ("5.0", "5.0 null", "5.0 IPP off")]
    for name, has in v["5.0"]["api"].items():
        yield f"| `{name}` | " + " | ".join("yes" if x else "no" for x in [has, *(v[o]["api"][name] for o in others)]) + " |"


def builds_figures() -> list[str]:
    """The phrases the section's builds paragraphs take from opencv_builds.json, after checking what they claim."""
    doc = load("opencv_builds.json")
    v, c = doc["variants"], doc["config"]
    old, ipp, null = (v[k]["stages"] for k in ("4.14.0.94", "5.0 IPP off", "5.0 null"))
    worst_null = max(null, key=lambda s: null[s]["vs_5.0"]["ci"][1])
    big = {"gaze", "gaze_oblique", "nanodet"}
    assert all(old[s]["vs_5.0"]["ci"][0] > null[s]["vs_5.0"]["ci"][1] for s in big)  # outside the null's reach
    assert all(0.85 <= old[s]["vs_5.0"]["ratio"] <= 1.15 for s in set(old) - big)
    assert {s for s in old if not old[s]["same_output"]} == {"gaze", "gaze_oblique"}
    assert {s for s in ipp if not ipp[s]["same_output"]} == {"saliency"}
    assert all(ipp[s]["vs_5.0"]["ci"][0] <= null[s]["vs_5.0"]["ci"][1] for s in ipp)
    assert v["4.14.0.94"]["cv2"] == "4.14.0" and v["5.0"]["cv2"] == "5.0.0" and not v["5.0 IPP off"]["ipp"]
    return [f"load average {doc['provenance']['load_avg'][0]} at the start and {doc['load_avg_end'][0]} at the end",
            f"{c['rounds']} rounds", f"each stage {c['repeats']} times",
            f"reaches x{null[worst_null]['vs_5.0']['ci'][1]:.2f} (`{worst_null}`)",
            f"{iv(old['gaze']['vs_5.0'])} of its time on the straight 512-px gaze and {iv(old['gaze_oblique']['vs_5.0'])}",
            f"{iv(old['nanodet']['vs_5.0'])} for 4.14",
            f"({v['5.0']['stages']['gaze']['p50_ms']:.1f} ms here against {v['5.0']['stages']['segment']['p50_ms']:.1f}"]


def test_shipped_flags_follow_the_rule():
    doc = load("opencv5_features.json")
    assert {f: getattr(segment, f) for f in feat.FLAGS} == {f: adopted(doc, f) for f in feat.FLAGS}
    assert segment.DNN_ENGINE == "new" and doc["config"]["onnx"]  # the v1 parts ran
    assert all(i["identical"]["null"] == i["photos"] for i in doc["identity"].values())  # the identity is not noise


def test_features_section_is_the_json():
    text = section()
    rows = [*identity_rows(), *timing_rows(), *profile_rows(), *builds_rows(), *api_rows()]
    assert [r for r in rows if " ".join(r.split()) not in text] == []


if __name__ == "__main__":
    print(*identity_rows(), *timing_rows(), *profile_rows(), *builds_rows(), *api_rows(), sep="\n")
