"""Validate the cleaned notebook format and preservation manifest; no model runs."""

import ast
import hashlib
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "docs/cleanup_manifest.json").read_text())

for item in manifest["notebooks"]:
    path = ROOT / item["path"]
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    hashes = [hashlib.sha256(cell.source.encode()).hexdigest() for cell in code_cells]
    if hashes != item["original_code_cell_sha256_in_order"]:
        raise ValueError(f"Code source or order differs from the preserved original: {path.name}")

    shell_cells = 0
    for index, cell in enumerate(code_cells, start=1):
        if cell.outputs or cell.execution_count is not None:
            raise ValueError(f"Saved execution output remains: {path.name}, code cell {index}")
        if any(line.lstrip().startswith(("!", "%")) for line in cell.source.splitlines()):
            # These Colab/IPython cells are preserved and covered by the hashes.
            shell_cells += 1
            continue
        ast.parse(cell.source, filename=f"{path.name}:code-cell-{index}")
    print(f"PASS {path.name}: {len(code_cells)} code cells preserved; "
          f"{shell_cells} IPython/shell cells excluded from Python AST parsing")

print("Static validation only. Full Colab execution and data/result audits were not performed.")
