"""Execute notebook 07 on synthetic fixtures only; no model fitting or downloads."""

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
work = ROOT / "work"
work.mkdir(exist_ok=True)
smoke_root = Path(tempfile.mkdtemp(prefix="mining-smoke-", dir=work))
fixture = runpy.run_path(str(ROOT / "tests/test_unknown_mining.py"))[
    "make_mining_fixture"
](smoke_root)

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
os.environ["SNOOPPI_ROOT"] = str(smoke_root)
os.environ["SNOOPPI_REPO_ROOT"] = str(ROOT)
os.environ["SNOOPPI_SPLIT_DIR"] = "splits"
os.environ.pop("SNOOPPI_CLUSTER_FILE", None)
os.environ["IPYTHONDIR"] = str(smoke_root / "ipython")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

source = ROOT / "drafting_code/07_mine_unknown_negatives.ipynb"
# First exercise safe defaults: only split/unknown inputs, no checkpoint/cache
# loading and no mining directory creation. An intentionally nonexistent cache
# path makes accidental default inference fail immediately.
audit_only = nbformat.read(source, as_version=4)
for cell in audit_only.cells:
    if cell.cell_type == "code":
        cell.source = cell.source.replace(
            'UNKNOWN_CSV = PATHS["root"] / "raw" / "snooppi_unknown.csv"',
            f'UNKNOWN_CSV = Path({str(fixture["unknown"])!r})',
        )
        cell.source = cell.source.replace(
            'PROTT5_CACHE_PATHS = [PLM_CACHE_FILES["ProtT5"]]',
            'PROTT5_CACHE_PATHS = [Path("/nonexistent/disabled-cache.pt")]',
        )
NotebookClient(
    audit_only,
    timeout=120,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute()
assert not (smoke_root / "mined").exists()

notebook = nbformat.read(source, as_version=4)
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    for switch in (
        "CHECK_CACHE_COVERAGE",
        "CHECKPOINT_PROVENANCE_REVIEWED",
        "WRITE_AUDIT",
        "RUN_SCORING",
        "RUN_SELECTION",
        "CACHE_VERDICT",
    ):
        cell.source = cell.source.replace(f"{switch} = False", f"{switch} = True")
    cell.source = cell.source.replace(
        'UNKNOWN_CSV = PATHS["root"] / "raw" / "snooppi_unknown.csv"',
        f'UNKNOWN_CSV = Path({str(fixture["unknown"])!r})',
    )
    cell.source = cell.source.replace(
        'CHECKPOINT_PATH = PATHS["checkpoints"] / "best_predictor_checkpoint.pt"',
        f'CHECKPOINT_PATH = Path({str(fixture["checkpoint"])!r})',
    )
    cell.source = cell.source.replace(
        'PROTT5_CACHE_PATHS = [PLM_CACHE_FILES["ProtT5"]]',
        f'PROTT5_CACHE_PATHS = [Path({str(smoke_root / "cache.pt")!r})]',
    )
    cell.source = cell.source.replace(
        '"FILL_IN_HUGGING_FACE_COMMIT_OR_EXPORT_REVISION"', '"synthetic_fixture_only"'
    )
    cell.source = cell.source.replace("CHUNK_SIZE = 5_000", "CHUNK_SIZE = 3")
    cell.source = cell.source.replace(
        "INFERENCE_BATCH_SIZE = 512", "INFERENCE_BATCH_SIZE = 2"
    )
    cell.source = cell.source.replace(
        'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"', 'DEVICE = "cpu"'
    )

NotebookClient(
    notebook,
    timeout=120,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
).execute()
executed = smoke_root / "executed"
executed.mkdir()
nbformat.write(audit_only, executed / "07_audit_only.ipynb")
nbformat.write(notebook, executed / source.name)
html, _ = HTMLExporter().from_notebook_node(notebook)
(executed / source.with_suffix(".html").name).write_text(html)
verdict = json.loads((smoke_root / "mined/mining_verdict.json").read_text())
assert verdict["selected_pairs"] == 2
assert verdict["inputs"]["valid_scored_rows"] == 7
assert not verdict["training_performed"] and not verdict["binary_labels_assigned"]
print(
    "PASS: notebook 07 executed end to end; 7 synthetic scores, 2 candidate review pairs, no training."
)
print("Synthetic executed notebook and HTML:", executed)
