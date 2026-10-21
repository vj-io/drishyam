"""Build the Kaggle notebooks (NOTEBOOKS: one per variant) from the `# %%` cells of training/train_dacl10k.py.

    python training/make_notebook.py           # write the notebooks (after any change to the script or CARRIED)
    python training/make_notebook.py --check   # exit 1 if a committed notebook is stale (tests run this)

The variants differ in one line, the script's `VARIANT = "v2a"`, which selects a flag line of its VARIANTS.

The module docstring (before the first `# %%`) is left out; a `# %% [markdown]` cell becomes markdown without its
`# ` prefixes. The notebook has to run on Kaggle without the repo, so the script's empty `REPO_FILES` line is
replaced by the CARRIED files' text, which the setup cell writes to disk, and its empty `SOURCE_SHA256` line by
`source_sha256()`, which run_info.json records in place of a git SHA. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "training" / "train_dacl10k.py"
NOTEBOOK = ROOT / "training" / "drishyam_dacl10k_kaggle.ipynb"  # v2a: the one README's Kaggle steps import
NOTEBOOKS = {"v2a": NOTEBOOK, "v2b": ROOT / "training" / "drishyam_dacl10k_kaggle_v2b.ipynb"}
COLAB = {"v2c": ROOT / "training" / "drishyam_dacl10k_colab_v2c.ipynb"}  # the same cells behind COLAB_CELL
CARRIED = ("training/dacl_common.py", "training/class_map.json", "training/export_onnx.py", "training/parity.py",
           "drishyam/__init__.py", "drishyam/imaging.py", "drishyam/measure.py",
           "drishyam/segment.py")  # what the cells import or run
PLACEHOLDER = "REPO_FILES: dict[str, str] = {}\n"
SHA_PLACEHOLDER = 'SOURCE_SHA256 = ""'
VARIANT_LINE = 'VARIANT = "v2a"\n'
# Colab: checkpoints live on Google Drive (every epoch: last.pt, best.pt, metrics.json), the data on local disk.
# A session that dies is resumed by running all cells again: --resume auto picks up <Drive folder>/last.pt.
COLAB_CELL = """# Colab only: durable checkpoints on Google Drive, data on the fast local disk. After a disconnect, Run all again.
import importlib.util

from google.colab import drive

drive.mount("/content/drive")
CKPT = "/content/drive/MyDrive/drishyam_checkpoints/{VARIANT_NAME}"
os.makedirs(CKPT, exist_ok=True)
os.makedirs("/kaggle/working", exist_ok=True)  # switches on the setup cell below (pins, OpenCV 5 in its own folder)
if importlib.util.find_spec("kagglehub") is None:
    subprocess.run(["pip", "install", "-q", "kagglehub"], check=True)
import kagglehub

DATA = kagglehub.dataset_download("winterqi131/dacl10k-v2-devphase")  # local disk: never train over the Drive mount
os.environ["DACL_FLAGS"] = f"--data {DATA} --out {CKPT} --resume auto --hours 11"
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True,
                     check=False)
print(gpu.stdout, "checkpoints:", sorted(os.listdir(CKPT)))
"""


def source_sha256() -> str:
    """sha256 over the training script and the CARRIED files as they are on disk (the script with its placeholders)."""
    h = hashlib.sha256()
    for rel in ("training/train_dacl10k.py", *CARRIED):
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def build(variant: str = "v2a") -> str:
    src = SCRIPT.read_text()
    if src.count(PLACEHOLDER) != 1 or src.count(SHA_PLACEHOLDER) != 1 or src.count(VARIANT_LINE) != 1:
        raise SystemExit(f"{SCRIPT.name}: expected one line each of {PLACEHOLDER.strip()!r}, {SHA_PLACEHOLDER!r} and "
                         f"{VARIANT_LINE.strip()!r}")
    if f'\n    "{variant}": ' not in src:
        raise SystemExit(f"{SCRIPT.name}: no {variant!r} in VARIANTS")
    src = src.replace(VARIANT_LINE, f'VARIANT = "{variant}"\n')
    src = src.replace(SHA_PLACEHOLDER, f'SOURCE_SHA256 = "{source_sha256()}"')
    carried = "".join(f"    {rel!r}: {(ROOT / rel).read_text()!r},\n" for rel in CARRIED)
    src = src.replace(PLACEHOLDER, "REPO_FILES = {  # written by make_notebook.py from the repo: edit those files\n"
                      + carried + "}\n")
    cells = []
    for i, block in enumerate(src.split("\n# %%")[1:]):
        head, _, body = block.partition("\n")
        if head.strip() == "[markdown]":
            text = "\n".join(ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in body.strip().splitlines())
            cells.append({"cell_type": "markdown", "id": f"cell-{i}", "metadata": {}, "source": _lines(text)})
        else:
            code = f"#{head}\n{body.strip()}"
            cells.append({"cell_type": "code", "execution_count": None, "id": f"cell-{i}", "metadata": {},
                          "outputs": [], "source": _lines(code)})
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def build_colab(variant: str) -> str:
    """The variant's notebook with COLAB_CELL before its setup cell (Drive checkpoints, local data, auto-resume)."""
    nb = json.loads(build(variant))
    code = COLAB_CELL.replace("{VARIANT_NAME}", variant)
    at = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).startswith("# Setup"))
    nb["cells"].insert(at, {"cell_type": "code", "execution_count": None, "id": "colab-setup", "metadata": {},
                           "outputs": [], "source": _lines(code)})
    nb["metadata"]["accelerator"] = "GPU"
    nb["metadata"]["colab"] = {"provenance": [], "gpuType": "T4"}
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="compare with the committed notebooks, write nothing")
    a = p.parse_args(argv)
    stale = 0
    jobs = [(v, p, build) for v, p in NOTEBOOKS.items()] + [(v, p, build_colab) for v, p in COLAB.items()]
    for variant, path, make in jobs:
        text = make(variant)
        if not a.check:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)}")
        elif not (path.exists() and path.read_text() == text):
            print(f"{path.relative_to(ROOT)} is stale: run python training/make_notebook.py", file=sys.stderr)
            stale = 1
    return stale


if __name__ == "__main__":
    sys.exit(main())
