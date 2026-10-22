import itertools
import struct
import zlib

import cv2
import numpy as np
import pytest

from drishyam import imaging

from .conftest import concrete


def png_header(w, h):
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + chunk + struct.pack(">I", zlib.crc32(chunk))


def test_sniff_reads_jpeg_and_png_sizes():
    img = np.zeros((123, 321, 3), np.uint8)
    for ext in (".jpg", ".png"):
        _ok, buf = cv2.imencode(ext, img)
        assert imaging.sniff_size(buf.tobytes()) == (321, 123)


def test_rejects_non_images_and_bombs():
    with pytest.raises(imaging.ImageError):
        imaging.decode_image(b"GIF89a....")
    with pytest.raises(imaging.ImageError):
        imaging.decode_image(b"")
    # 60000 x 60000 header with no pixel data: refused before any decode is attempted.
    with pytest.raises(imaging.ImageError, match="limit"):
        imaging.decode_image(png_header(60000, 60000))
    with pytest.raises(imaging.ImageError, match="minimum"):
        imaging.decode_image(png_header(10, 10))
    _ok, buf = cv2.imencode(".jpg", np.zeros((200, 200, 3), np.uint8))
    with pytest.raises(imaging.ImageError):
        imaging.decode_image(buf.tobytes()[:300])  # truncated


def test_decode_roundtrip():
    img = (np.random.default_rng(0).random((100, 150, 3)) * 255).astype(np.uint8)
    _ok, buf = cv2.imencode(".png", img)
    assert np.array_equal(imaging.decode_image(buf.tobytes()), img)


def test_pyramid_levels():
    pyr = imaging.build_pyramid(np.zeros((3000, 4000, 3), np.uint8))
    assert [p.shape[:2] for p in pyr] == [(3000, 4000), (1500, 2000), (750, 1000), (375, 500), (188, 250)]


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_gaze_maps_back_to_l0_exactly(level):
    img = np.zeros((2048, 2048, 3), np.uint8)
    s = 2**level
    X, Y = 1024 + 40 * s, 1024 - 30 * s  # inside the window; a 2s block survives pyrDown
    img[Y - s : Y + s, X - s : X + s] = 255
    pyr = imaging.build_pyramid(img)
    crop, to_l0 = imaging.gaze(pyr, 1024, 1024, level, size=256)
    v, u = np.unravel_index(np.argmax(crop[..., 0]), crop.shape[:2])
    (x, y), = imaging.map_points(to_l0, [[u, v]])
    assert abs(x - X) <= s and abs(y - Y) <= s
    # Pure translation: the crop's centre is the requested L0 point.
    (cx, cy), = imaging.map_points(to_l0, [[128, 128]])
    assert (cx, cy) == (1024, 1024)


def test_footprint_marks_the_crop_pixels_that_show_the_image():
    pyr = imaging.build_pyramid(np.zeros((600, 800, 3), np.uint8))
    _, to_l0 = imaging.gaze(pyr, 400, 300, 1, size=512)  # L0 window [-112, -212, 1024, 1024]
    valid = imaging.footprint(to_l0, (800, 600), 512)
    ys, xs = np.nonzero(valid)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (56, 455, 106, 405)  # x = 2u - 112 in [0, 799]
    _, tilted = imaging.gaze(pyr, 400, 300, 0, yaw=20, pitch=-10, size=256)  # the general (homography) path
    inside = cv2.warpPerspective(np.ones((600, 800), np.uint8), tilted, (256, 256),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) > 0
    assert np.mean(imaging.footprint(tilted, (800, 600), 256) == inside) > 0.995


def test_gaze_yaw_pitch_changes_view():
    pyr = imaging.build_pyramid((np.random.default_rng(1).random((1024, 1024, 3)) * 255).astype(np.uint8))
    straight, _ = imaging.gaze(pyr, 512, 512, 0, size=128)
    tilted, _ = imaging.gaze(pyr, 512, 512, 0, yaw=15, pitch=-10, size=128)
    assert straight.shape == tilted.shape and not np.array_equal(straight, tilted)


def test_quality_gate():
    sharp = (np.random.default_rng(2).random((256, 256, 3)) * 255).astype(np.uint8)
    assert imaging.quality(sharp)["ok"]
    blurred = cv2.GaussianBlur(np.full((256, 256, 3), 128, np.uint8), (31, 31), 10)
    assert imaging.quality(blurred)["reason"] == "blur"
    glare = sharp.copy()
    glare[:128] = 255
    assert imaging.quality(glare)["reason"] == "glare"


def test_overview_finds_the_salient_spot(scene):
    pyr = imaging.build_pyramid(scene)
    sal = imaging.saliency(pyr[3])
    cands = imaging.find_candidates(sal, 3, 1024, (scene.shape[1], scene.shape[0]))
    assert 1 <= len(cands) <= 20
    assert [c.id for c in cands] == [f"r{i + 1}" for i in range(len(cands))]
    assert all(0 <= c.prior <= 1 for c in cands)
    # The rebar spall (around x=1500, y=450) is the most salient thing in the scene.
    assert any(abs(c.cx - 1500) < 400 and abs(c.cy - 450) < 400 for c in cands[:3])


def test_sound_concrete_is_not_salient_but_glare_still_gets_a_look():
    """The absolute null: a sound surface's texture is not stretched to full scale (with plain p99.5 scaling about
    a quarter of this overview scored >= 0.5, the candidate threshold). A blown-out patch stays salient, so the
    quality gate gets to refuse it and the work order marks it not assessed."""
    img = concrete(1536, 2048, 0)
    assert (imaging.saliency(imaging.build_pyramid(img)[3]) >= 0.5).mean() < 0.01
    cv2.circle(img, (1000, 700), 120, (255, 255, 255), -1)
    cands = imaging.find_candidates(imaging.saliency(imaging.build_pyramid(img)[3]), 3, 1024, (2048, 1536))
    assert abs(cands[0].cx - 1000) < 150 and abs(cands[0].cy - 700) < 150


def test_decoded_bytes_budget():
    """A small 16-bit RGBA PNG header claiming 25 MP would make libpng allocate ~200 MB: refused."""
    import struct as st

    def png16(w, h, colour=6, depth=16):
        ihdr = st.pack(">IIBBBBB", w, h, depth, colour, 0, 0, 0)
        chunk = b"IHDR" + ihdr
        return b"\x89PNG\r\n\x1a\n" + st.pack(">I", 13) + chunk + st.pack(">I", zlib.crc32(chunk))

    with pytest.raises(imaging.ImageError, match="decode to"):
        imaging.decode_image(png16(5000, 5000))
    assert imaging.sniff(png16(100, 100))[2] == 8.0
    assert imaging.sniff(png16(100, 100, colour=2, depth=8))[2] == 3.0
    with pytest.raises(imaging.ImageError, match="colour type"):
        imaging.sniff(png16(100, 100, colour=5))
    sixteen = (np.random.default_rng(3).random((300, 300, 3)) * 65535).astype(np.uint16)
    assert imaging.decode_image(cv2.imencode(".png", sixteen)[1].tobytes()).dtype == np.uint8


def test_l0_box_is_never_degenerate_under_yaw_and_pitch():
    pyr = imaging.build_pyramid(np.zeros((1536, 2048, 3), np.uint8))
    for x, y, yaw, pitch in ((0, 768, -40, -40), (2047, 0, 30, 20), (1024, 1535, 40, -30)):
        _, to_l0 = imaging.gaze(pyr, x, y, 0, yaw=yaw, pitch=pitch, size=256)
        for rect in ((0, 0, 256, 256), (10, 200, 40, 30), (200, 10, 30, 40)):
            bx, by, bw, bh = imaging.l0_box(rect, to_l0, (2048, 1536))
            assert bw >= 0 and bh >= 0 and 0 <= bx <= 2048 and 0 <= by <= 1536
            assert bx + bw <= 2048 and by + bh <= 1536


# ---- the blur gate (JobConfig.blur_gate): the Laplacian variance, or the re-blur ratio (imaging.reblur) ----------
def dead_leaves(n: int, seed: int = 0) -> np.ndarray:
    """Grey discs of power-law sizes drawn over each other: the dead-leaves model, whose statistics do not change with
    scale (Lee, Mumford and Huang 2001), so its pyramid levels are the same scene at half and quarter scale."""
    rng = np.random.default_rng(seed)
    img = np.full((n, n, 3), 128, np.uint8)
    for _ in range(6000):
        g = int(rng.integers(40, 220))
        cv2.circle(img, (int(rng.integers(0, n)), int(rng.integers(0, n))),
                   int(n * 0.2 / (1 + rng.pareto(1.5) * 30)) + 1, (g, g, g), -1, cv2.LINE_AA)
    return img


def central_views(img, levels=(0, 1, 2)) -> list[np.ndarray]:
    """The 512-px view at the photo's centre at each level, as the agent's gaze renders it."""
    pyr = imaging.build_pyramid(img)
    return [imaging.gaze(pyr, img.shape[1] / 2, img.shape[0] / 2, k)[0] for k in levels]


def gray(view):
    return cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)


def cracked_concrete():
    from .conftest import add_crack

    return add_crack(concrete(2048, 2048, 3), (200, 900))


@pytest.mark.parametrize("make", [cracked_concrete, lambda: dead_leaves(2048)])
def test_the_reblur_ratio_does_not_move_with_contrast(make):
    """A quarter of the contrast (about the mean): the Laplacian variance falls 16-fold, so the Laplacian gate refuses
    the faint copy of a view it passes; the re-blur ratio is a ratio of gradient sums and stays within 8-bit rounding,
    so the re-blur gate decides the faint copy as it decides the view."""
    view = central_views(make(), (1,))[0]
    low = np.clip(view.mean() + (view.astype(np.float32) - view.mean()) * 0.25, 0, 255).astype(np.uint8)
    assert abs(imaging.reblur(gray(low)) - imaging.reblur(gray(view))) < 0.02
    assert imaging.quality(view)["ok"] and imaging.quality(low)["reason"] == "blur"
    assert imaging.quality(low, "reblur")["reason"] == imaging.quality(view, "reblur")["reason"]


def test_the_reblur_ratio_does_not_move_with_the_pyramid_level():
    """The same scale-free scene at L0, L1 and L2 (one and two pyrDown steps): the Laplacian variance spreads over more
    than 2x, the re-blur ratio over less than 0.05, a sixth of what a sigma-2 blur adds to it."""
    views = central_views(dead_leaves(2048))
    lap = [imaging.quality(v)["sharpness"] for v in views]
    rb = [imaging.reblur(gray(v)) for v in views]
    assert max(lap) > 2 * min(lap) and max(rb) - min(rb) < 0.05
    assert min(imaging.reblur(gray(cv2.GaussianBlur(v, (0, 0), 2.0))) - r for v, r in zip(views, rb)) > 0.3


@pytest.mark.parametrize("make", [cracked_concrete, lambda: dead_leaves(2048)])
def test_the_reblur_ratio_rises_with_blur_and_the_gate_refuses_a_sigma_2_view(make):
    """Monotone in the Gaussian sigma (view px), and a sigma-2 copy, the rule's 'too blurred to trust'
    (results/dev/blur_gate.json), is refused. (The cracked concrete as it is passes; the dead-leaves discs, flat
    between anti-aliased edges, read 0.41 and do not.)"""
    view = central_views(make(), (1,))[0]
    rb = [imaging.reblur(gray(cv2.GaussianBlur(view, (0, 0), s) if s else view)) for s in (0, 0.5, 1, 1.5, 2, 3, 4)]
    assert all(a < b for a, b in itertools.pairwise(rb)), rb
    assert imaging.quality(cv2.GaussianBlur(view, (0, 0), 2.0), "reblur")["reason"] == "blur"


def test_known_limit_a_faint_surface_blurred_past_its_grain_reads_sharp():
    """Known limit (results/dev/blur_gate.json): concrete grain at a fifth of its contrast, blurred by sigma 8 L0 px
    and seen at L1 (4 view px of blur). The grain is gone; what is left is 8-bit steps, which a measure that ignores
    contrast reads as sharp, so the re-blur gate passes the view. The Laplacian gate refuses it."""
    g = concrete(2048, 2048, 4).astype(np.float32)
    faint = np.clip(g.mean() + (g - g.mean()) * 0.2, 0, 255).astype(np.uint8)
    view = central_views(cv2.GaussianBlur(faint, (0, 0), 8.0), (1,))[0]
    assert imaging.quality(view, "reblur")["ok"] and imaging.quality(view)["reason"] == "blur"


def test_a_view_with_no_gradient_is_blur_under_both_gates_without_a_warning():
    """Nothing to judge focus by: the ratio's denominator is 0, so the measure is 1 (refused, as the Laplacian gate
    refuses it); no division warning, also on a view too small to keep any pixel inside the measure's border."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for flat in (np.full((512, 512, 3), 128, np.uint8), np.full((3, 3, 3), 128, np.uint8)):
            assert imaging.reblur(gray(flat)) == 1.0
            q = imaging.quality(flat, "reblur")
            assert (q["reason"], q["reblur"]) == ("blur", 1.0) and imaging.quality(flat)["reason"] == "blur"


def test_the_blur_gate_must_be_one_it_knows():
    with pytest.raises(ValueError, match="blur gate"):
        imaging.quality(np.zeros((64, 64, 3), np.uint8), "sharpness")


def without_blur_gate(monkeypatch):
    """drishyam.imaging without its lines marked `# blur-gate`: the quality gate as it was before the knob."""
    import re
    import sys
    import types
    from pathlib import Path

    src = Path(imaging.__file__).read_text()
    kept = "".join(ln for ln in src.splitlines(keepends=True) if not re.search(r"#\s*blur-gate\b", ln))
    assert kept != src
    mod = types.ModuleType("drishyam._imaging_without_blur_gate")
    mod.__package__ = "drishyam"
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    exec(compile(kept, "drishyam/imaging.py without the blur gate", "exec"), mod.__dict__)  # noqa: S102 -- our own source
    assert "gate" in imaging.quality.__code__.co_varnames and "gate" not in mod.quality.__code__.co_varnames
    return mod


def test_with_the_laplacian_gate_quality_is_what_it_was_on_every_bundled_sample(monkeypatch):
    """Every bundled sample, views at L0-L3 at its centre and its four quarter points: quality() with no gate and with
    "laplacian" returns exactly what the quality gate returned before the knob."""
    from pathlib import Path

    old = without_blur_gate(monkeypatch)
    samples = sorted((Path(__file__).resolve().parent.parent / "public" / "samples").glob("*.jpg"))
    assert len(samples) == 10
    n = 0
    for p in samples:
        img = cv2.imread(str(p))
        pyr, (h, w) = imaging.build_pyramid(img), img.shape[:2]
        for level in range(4):
            for fx, fy in ((0.5, 0.5), (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)):
                crop = imaging.gaze(pyr, fx * w, fy * h, level)[0]
                want = old.quality(crop)
                assert imaging.quality(crop) == want and imaging.quality(crop, "laplacian") == want
                n += 1
    assert n == 200
