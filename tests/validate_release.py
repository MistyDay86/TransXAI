#!/usr/bin/env python3
"""Lightweight integrity checks for the public TransXAI release."""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(actual: str, expected: float, tolerance: float = 5e-7) -> bool:
    return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=tolerance)


def main() -> None:
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    require(len(scripts) == 13, f"Expected 13 public scripts, found {len(scripts)}")
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    json_files = sorted((ROOT / "configs").glob("*.json")) + sorted(
        (ROOT / "results" / "manifests").glob("*.json")
    )
    require(json_files, "No JSON contracts found")
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))

    overall = load_csv(
        "results/tables/backbone_replication/table_1_backbone_budget_overall.csv"
    )
    require(len(overall) == 4, "Backbone-by-budget summary must contain four rows")
    keyed = {(row["backbone"], float(row["budget"])): row for row in overall}
    expected = {
        ("resnet18", 0.1): 0.124973,
        ("resnet18", 0.2): 0.129406,
        ("regnet_x_400mf", 0.1): 0.111677,
        ("regnet_x_400mf", 0.2): 0.115755,
    }
    require(set(keyed) == set(expected), "Unexpected backbone/budget keys")
    for key, value in expected.items():
        require(close(keyed[key]["delta_cpts"], value), f"Delta CPTS mismatch for {key}")

    cells = load_csv(
        "results/tables/backbone_replication/table_s1_all_360_cells.csv"
    )
    require(len(cells) == 360, f"Expected 360 cells, found {len(cells)}")

    quality = {
        row["check"]: row
        for row in load_csv(
            "results/tables/backbone_replication/table_s2_quality_controls.csv"
        )
    }
    require(quality["Paired donor rows"]["value"] == "46080", "Donor-row count mismatch")
    require(quality["Analysis cells"]["value"] == "360", "Cell count mismatch")
    require(quality["Positive cell means"]["value"] == "360", "Positive-cell gate failed")
    require(quality["Exact-k violations"]["value"] == "0", "Exact-k violations present")
    require(
        quality["Local-faithfulness violations"]["value"] == "0",
        "Local-faithfulness violations present",
    )

    forbidden = ["efficientnet", "learned_explainers_transxai", "05_train_learned"]
    public_paths = "\n".join(str(path.relative_to(ROOT)).lower() for path in ROOT.rglob("*"))
    for token in forbidden:
        require(token not in public_paths, f"Obsolete release artifact found: {token}")

    print(
        f"PASS: {len(scripts)} scripts, {len(json_files)} JSON contracts, "
        "360 cells, and frozen headline results validated."
    )


if __name__ == "__main__":
    main()
