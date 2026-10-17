# Bundled models (third-party, permissive licences)

| File | Source | Licence | sha256 |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo, YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | MIT (`LICENSE.yunet`) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `object_detection_nanodet_2022nov.onnx` | [OpenCV Zoo, NanoDet-Plus](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_nanodet) | Apache 2.0 (`LICENSE.nanodet`) | `4b82da9944b88577175ee23a459dce2e26e6e4be573def65b1055dc2d9720186` |

Both are used only by `drishyam/privacy.py` to blur faces, people and vehicles
before an image is analysed or displayed. Fonts in `public/fonts/` are SIL OFL 1.1
(`public/fonts/OFL-plex.txt`, `public/fonts/OFL-archivo.txt`).
