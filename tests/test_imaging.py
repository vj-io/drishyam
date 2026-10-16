import struct
import zlib

import cv2
import numpy as np
import pytest

from drishyam import imaging


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
