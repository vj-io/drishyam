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
