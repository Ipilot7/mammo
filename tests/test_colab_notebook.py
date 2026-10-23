import json
import re
from pathlib import Path

from scripts.build_colab_notebook import RUN_HELPER, pack

ROOT = Path(__file__).resolve().parents[1]


def test_colab_notebook_embeds_current_canonical_source():
    notebook = json.loads((ROOT / "train" / "colab_vscode.ipynb").read_text())
    code_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and 'CODE = "' in "".join(cell["source"])
    )
    embedded = re.search(r'CODE = "([A-Za-z0-9+/=]+)"', code_cell)

    assert embedded is not None
    assert embedded.group(1) == pack()


def test_colab_upload_publishes_sha256_sidecar():
    assert "hashlib.sha256" in RUN_HELPER
    assert "path_in_repo=f'{name}.sha256'" in RUN_HELPER
