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


def test_a_finer_overview_keeps_a_one_pixel_line_through_the_downsample():
    """saliency(img, fine): edges and black-hat from one level finer, max-pooled, keep a 1-px line that pyrDown blurs;
    sound concrete stays below the candidate threshold either way (the floors are in the same units)."""
    img = concrete(1536, 2048, 0)
    pyr = imaging.build_pyramid(img)
    assert (imaging.saliency(pyr[3], pyr[2]) >= 0.5).mean() < 0.01
    fine = pyr[2].copy()
    cv2.line(fine, (100, 100), (400, 300), (110, 110, 112), 1)
    line = imaging._max_pool(np.all(fine == (110, 110, 112), 2).astype(np.float32), pyr[3].shape[:2]) > 0
    assert line.sum() >= 150  # every 2x2 block the 301-px line crosses keeps it
    smooth = cv2.pyrDown(fine)
    # the share of the line's overview pixels at the candidate threshold: 0.16 against 0.04 from the overview alone
    assert (imaging.saliency(smooth, fine)[line] >= 0.5).mean() > 2 * (imaging.saliency(smooth)[line] >= 0.5).mean()


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
