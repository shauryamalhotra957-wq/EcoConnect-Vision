"""Validate the committed notebook and model artifact without loading the model."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "biodeg.ipynb"
MODEL = ROOT / "waste_scanner_model.keras"


def validate_notebook(path: Path) -> tuple[int, int]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"missing notebook: {path.relative_to(ROOT)}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid notebook JSON: {exc}") from exc

    if document.get("nbformat") != 4:
        raise SystemExit("biodeg.ipynb must use notebook format 4")

    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SystemExit("biodeg.ipynb must contain at least one cell")

    code_cells = 0
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise SystemExit(f"cell {index} must be an object")
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown", "raw"}:
            raise SystemExit(f"cell {index} has unsupported type: {cell_type!r}")
        source = cell.get("source")
        if not isinstance(source, (str, list)):
            raise SystemExit(f"cell {index} source must be text or a list of lines")
        if cell_type == "code":
            code_cells += 1

    if code_cells == 0:
        raise SystemExit("biodeg.ipynb must contain at least one code cell")

    return len(cells), code_cells


def validate_model(path: Path) -> int:
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise SystemExit(f"missing model artifact: {path.relative_to(ROOT)}") from exc

    if size < 1024:
        raise SystemExit("waste_scanner_model.keras is unexpectedly small")

    return size


def main() -> None:
    cell_count, code_cell_count = validate_notebook(NOTEBOOK)
    model_size = validate_model(MODEL)
    print(
        "repository validation passed: "
        f"{cell_count} notebook cells ({code_cell_count} code), "
        f"{model_size} model bytes"
    )


if __name__ == "__main__":
    main()
