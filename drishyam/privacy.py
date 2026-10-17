"""Privacy blur and the surveillance guard (FR-19).

Faces come from YuNet (`cv2.FaceDetectorYN`, MIT) and people and vehicles from
NanoDet-Plus (Apache 2.0), both from OpenCV Zoo and both run on OpenCV DNN. The
blur happens before any crop is stored, displayed or analysed. Drishyam
inspects structures, not people: an image mostly filled by a person is refused.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "models"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
NANODET = MODELS / "object_detection_nanodet_2022nov.onnx"
PERSON_FRAC_MAX = 0.25
DETECT_SIDE = 640
NANO_SIZE = 416
NANO_REG_MAX = 7
PERSON, VEHICLES = 0, (1, 2, 3, 5, 7)  # COCO: bicycle, car, motorcycle, bus, truck


class PeopleDominant(ValueError):
    """The photo is of a person, not a structure."""


class _Detectors:
    def __init__(self) -> None:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # the new DNN engine warns about targets it ignores
        self.face = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8, 0.3, 50)
        self.nano = cv2.dnn.readNet(str(NANODET))

    def faces(self, img: np.ndarray) -> list[list[int]]:
        h, w = img.shape[:2]
        self.face.setInputSize((w, h))
        _, out = self.face.detect(img)
        return [] if out is None else [[int(v) for v in f[:4]] for f in out]

    def objects(self, img: np.ndarray, threshold: float = 0.45) -> list[tuple[int, list[int]]]:
        """NanoDet on a letterboxed 416 x 416 input; boxes come back in `img` pixels."""
        h, w = img.shape[:2]
        k = NANO_SIZE / max(h, w)
        canvas = np.zeros((NANO_SIZE, NANO_SIZE, 3), np.float32)
        canvas[: round(h * k), : round(w * k)] = cv2.resize(img, (round(w * k), round(h * k)))
        canvas = (canvas - (103.53, 116.28, 123.675)) / (57.375, 57.12, 58.395)
        self.nano.setInput(cv2.dnn.blobFromImage(canvas.astype(np.float32)))
        outs = self.nano.forward(self.nano.getUnconnectedOutLayersNames())
        # Output order differs between DNN engines, so pair heads by shape: 80 class scores
        # and 4 x 8 distance bins per anchor, one grid per stride.
        by_size = lambda last: sorted((o.reshape(-1, last) for o in outs if o.shape[-1] == last), key=len)
        boxes, scores = [], []
        for cls, reg in zip(by_size(80), by_size(4 * (NANO_REG_MAX + 1)), strict=True):
            n = round(len(cls) ** 0.5)
            stride = NANO_SIZE // n
            xv, yv = np.meshgrid(np.arange(n) * stride, np.arange(n) * stride)
            anchors = np.column_stack([xv.ravel(), yv.ravel()]) + 0.5 * (stride - 1)
            reg = reg.reshape(-1, NANO_REG_MAX + 1)
            e = np.exp(reg - reg.max(axis=1, keepdims=True))
            dist = ((e / e.sum(axis=1, keepdims=True)) @ np.arange(NANO_REG_MAX + 1)).reshape(-1, 4) * stride
            boxes.append(np.column_stack([anchors - dist[:, :2], anchors + dist[:, 2:]]))
            scores.append(cls)
        b, s = np.concatenate(boxes), np.concatenate(scores)
        ids, conf = s.argmax(axis=1), s.max(axis=1)
        xywh = np.column_stack([b[:, :2], b[:, 2:] - b[:, :2]]) / k
        keep = cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), threshold, 0.6)
        return [(int(ids[i]), [int(v) for v in xywh[i]]) for i in np.array(keep).ravel()
                if ids[i] == PERSON or ids[i] in VEHICLES]


_DET: _Detectors | None = None


def detectors() -> _Detectors:
    global _DET
    if _DET is None:
        _DET = _Detectors()
    return _DET


def blur_boxes(img: np.ndarray, boxes: list[list[int]]) -> None:
    H, W = img.shape[:2]
    for x, y, w, h in boxes:
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = cv2.GaussianBlur(img[y0:y1, x0:x1], (0, 0), max(6.0, (x1 - x0) / 6))


def privacy_pass(img: np.ndarray, max_person_frac: float = PERSON_FRAC_MAX) -> dict:
    """Detect on a small copy, blur faces, people and vehicles in place at full resolution."""
    H, W = img.shape[:2]
    k = min(1.0, DETECT_SIDE / max(H, W))
    small = cv2.resize(img, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA) if k < 1 else img
    det = detectors()
    up = lambda b: [round(b[0] / k), round(b[1] / k), round(b[2] / k), round(b[3] / k)]
    faces = [up(b) for b in det.faces(small)]
    objects = [(c, up(b)) for c, b in det.objects(small)]
    people = [b for c, b in objects if c == PERSON]
    biggest = max((b[2] * b[3] for b in people), default=0) / (W * H)
    if biggest > max_person_frac:
        raise PeopleDominant(f"a person fills {biggest:.0%} of the frame; Drishyam inspects structures, not people")
    boxes = faces + [b for _, b in objects]
    blur_boxes(img, boxes)
    return {"faces_blurred": len(faces), "people_blurred": len(people),
            "vehicles_blurred": len(objects) - len(people), "boxes": boxes}
