"""Validate draft notebook schema, Python syntax, shared setup and output defaults."""

import ast
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
setup = None
for path in sorted((ROOT / "drafting_code").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    current_setup = notebook.cells[2].source
    if setup is None:
        setup = current_setup
    elif current_setup != setup:
        raise ValueError(f"Inconsistent shared setup: {path.name}")
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{path.name}:cell-{i}")
            if cell.outputs or cell.execution_count is not None:
                raise ValueError(
                    f"Draft contains stale execution outputs: {path.name}:cell-{i}"
                )
            for switch in (
                "WRITE_OUTPUTS",
                "WRITE_SPLITS",
                "WRITE_NEGATIVE_CANDIDATES",
                "TRAIN_MODEL",
                "RUN_SCORING",
                "WRITE_AUDIT",
                "RUN_SELECTION",
                "CACHE_VERDICT",
                "CHECKPOINT_PROVENANCE_REVIEWED",
                "ENABLE_LORA_SETUP",
            ):
                if f"{switch} = True" in cell.source:
                    raise ValueError(f"Enabled action default: {path.name}: {switch}")
    print(
        f"PASS {path.name}: schema, AST, shared setup, cleared outputs, disabled action switches"
    )
for path in (ROOT / "drafting_code").glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
print(
    "Static checks only. Use tests and synthetic smoke execution for behavioral validation."
)
