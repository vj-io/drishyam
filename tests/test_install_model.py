"""scripts/install_model.sh's model-card writer (its second Python heredoc), run on copies in a temporary folder."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_reinstall_keeps_hand_recorded_provenance_and_leaves_readme_alone(tmp_path):
    prov_file = ROOT / "models" / "drishyam_dacl10k.provenance.json"
    prov = json.loads(prov_file.read_text())
    card = (ROOT / "models" / "README.md").read_text()
    assert prov["known"] in card and all(s in card for s in prov["sha256"].values())
    code = (ROOT / "scripts" / "install_model.sh").read_text().split("\"$PY\" - <<'EOF'\n")[1].split("\nEOF\n")[0]
    models = tmp_path / "models"
    models.mkdir()
    for f in ("README.md", "drishyam_dacl10k.metrics.json"):
        shutil.copy(ROOT / "models" / f, models / f)
    prov["sha256"]["drishyam_dacl10k.onnx"] = hashlib.sha256(b"same").hexdigest()
    (models / prov_file.name).write_text(json.dumps(prov))
    for weights, kept in ((b"same", True), (b"other", False)):  # same weights reinstalled, then different ones
        (models / "drishyam_dacl10k.onnx").write_bytes(weights)
        subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
        assert (prov["known"] in (models / "README.md").read_text()) == kept
    assert not (tmp_path / "README.md").exists()  # the top-level README is never read or written


def test_v2_models_get_their_own_section_beside_v1(tmp_path):
    """A recipe v2 install (MODEL_NAME=drishyam_dacl10k_v2a) adds its own models/README.md section, chosen on the
    per-scale rule, and reinstalling v1 afterwards replaces only v1's section."""
    code = (ROOT / "scripts" / "install_model.sh").read_text().split("\"$PY\" - <<'EOF'\n")[1].split("\nEOF\n")[0]
    models = tmp_path / "models"
    models.mkdir()
    for f in ("README.md", "drishyam_dacl10k.metrics.json", "drishyam_dacl10k.provenance.json"):
        shutil.copy(ROOT / "models" / f, models / f)
    (models / "drishyam_dacl10k.onnx").write_bytes(b"v1")
    name = "drishyam_dacl10k_v2a"
    (models / f"{name}.onnx").write_bytes(b"v2a")
    v = lambda m5: {"miou19": 0.1, "miou5": m5, "iou19": {}, "iou5": [m5] * 5}
    hist = [{"epoch": e, "val_miou19": 0.3, "val_miou5": w, "iou5": [w] * 5, "iou19": {"Crack": 0.2, "ACrack": 0.4},
             "scales": {"512": v(s), "1024": v(s), "2048": v(s), "whole": v(w)}, "scale_mean": s, "select": "scales"}
            for e, (w, s) in enumerate([(0.40, 0.30), (0.35, 0.33)])]  # whole-photo best is epoch 0, per-scale epoch 1
    (models / f"{name}.metrics.json").write_text(json.dumps(hist))
    info = {"python": "3.12", "versions": {"torch": "2.9"}, "git_sha": None, "source_sha256": "ab",
            "flags": {"recipe": "v2", "variant": "v2a", "arch": "unet", "encoder": "tu-mobilenetv3_large_100"},
            "select_rule": "best.pt: the epoch with the highest scale_mean"}
    (models / f"{name}.run_info.json").write_text(json.dumps([info]))
    env = {"PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, env={**env, "MODEL_NAME": name})
    card = (models / "README.md").read_text()
    v2a = card.split(f"## Trained segmenter: `{name}`")[1]
    assert "## Trained segmenter: `drishyam_dacl10k`" in card.split(name)[0]  # v1's section is still there, first
    assert "Epoch 1 of 2" in v2a and "**0.330**" in v2a and "recipe v2, variant v2a" in v2a
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True, env=env)  # v1 again
    card = (models / "README.md").read_text()
    assert card.count("## Trained segmenter: `drishyam_dacl10k`") == 1 and card.count(f"## Trained segmenter: `{name}`") == 1
