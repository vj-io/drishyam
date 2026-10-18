"""Build training/drishyam_dacl10k_kaggle.ipynb from the `# %%` cells of training/train_dacl10k.py.

    python training/make_notebook.py           # write the notebook (after any change to the script or CARRIED)
    python training/make_notebook.py --check   # exit 1 if the committed notebook is stale (tests run this)

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
NOTEBOOK = ROOT / "training" / "drishyam_dacl10k_kaggle.ipynb"
CARRIED = ("training/dacl_common.py", "training/class_map.json", "training/export_onnx.py", "training/parity.py",
           "drishyam/__init__.py", "drishyam/measure.py", "drishyam/segment.py")  # what the cells import or run
PLACEHOLDER = "REPO_FILES: dict[str, str] = {}\n"
SHA_PLACEHOLDER = 'SOURCE_SHA256 = ""'


def source_sha256() -> str:
    """sha256 over the training script and the CARRIED files as they are on disk (the script with its placeholders)."""
    h = hashlib.sha256()
    for rel in ("training/train_dacl10k.py", *CARRIED):
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def build() -> str:
    src = SCRIPT.read_text()
    if src.count(PLACEHOLDER) != 1 or src.count(SHA_PLACEHOLDER) != 1:
        raise SystemExit(f"{SCRIPT.name}: expected one line {PLACEHOLDER.strip()!r} and one {SHA_PLACEHOLDER!r}")
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="compare with the committed notebook, write nothing")
    a = p.parse_args(argv)
    text = build()
    if a.check:
        if NOTEBOOK.exists() and NOTEBOOK.read_text() == text:
            return 0
        print(f"{NOTEBOOK.relative_to(ROOT)} is stale: run python training/make_notebook.py", file=sys.stderr)
        return 1
    NOTEBOOK.write_text(text)
    print(f"wrote {NOTEBOOK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
