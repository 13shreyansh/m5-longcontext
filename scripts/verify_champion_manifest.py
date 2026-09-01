#!/usr/bin/env python3
"""Fail if the provisional champion manifest drifts from committed source."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "CHAMPION_MANIFEST.json"


def validate_manifest(manifest: dict, root: Path = ROOT) -> str:
    if manifest["schema_version"] != 1:
        raise AssertionError("unsupported champion manifest schema")
    rows = manifest["rows_1_13"]
    if rows["fresh_float32_passed"] != rows["fresh_float32_total"]:
        raise AssertionError("ordinary-row manifest is not fully passing")
    row_14 = manifest["row_14"]
    if row_14["qkv_tile"] != [32, 512, 256, 1, 8]:
        raise AssertionError("manifest QKV tile differs from promoted geometry")
    exact_status = row_14["current_route_exact_b32_status"]
    if exact_status not in {
        "pending_contention_controlled_protocol",
        "complete_contention_controlled_protocol",
    }:
        raise AssertionError("manifest has an unsupported final exact-row status")
    expected_manifest_status = {
        "pending_contention_controlled_protocol": (
            "provisional_pending_contention_controlled_exact_measurement"
        ),
        "complete_contention_controlled_protocol": (
            "final_contention_controlled_exact_measurement_complete"
        ),
    }[exact_status]
    if manifest["status"] != expected_manifest_status:
        raise AssertionError("champion status does not match exact-row status")
    if exact_status == "complete_contention_controlled_protocol":
        exact_seconds = row_14.get("current_route_exact_b32_seconds")
        if (
            not isinstance(exact_seconds, list)
            or len(exact_seconds) != 3
            or any(
                not isinstance(value, (int, float)) or value <= 0
                for value in exact_seconds
            )
        ):
            raise AssertionError("complete exact-row status requires three positive runs")
        expected = {
            "current_route_exact_b32_median_seconds": sorted(exact_seconds)[1],
            "current_route_exact_b32_min_seconds": min(exact_seconds),
            "current_route_exact_b32_max_seconds": max(exact_seconds),
            "current_route_exact_b32_max_over_min": max(exact_seconds)
            / min(exact_seconds),
        }
        for field, value in expected.items():
            if not math.isclose(
                float(row_14.get(field, math.nan)),
                value,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise AssertionError(f"manifest exact-row statistic drift: {field}")
    if manifest["verification"]["official_mfu_or_combined_score"] is not None:
        raise AssertionError("manifest must not invent an official score")
    expected_commands = {
        "release_manifest": ".venv/bin/python scripts/verify_release_manifest.py",
        "provenance": ".venv/bin/python scripts/verify_solution_provenance.py",
        "tests": "PYTHONPATH=. .venv/bin/python -m pytest -q",
        "final_row_14_protocol": (
            ".venv/bin/python scripts/run_case14_final_protocol.py"
        ),
    }
    if manifest.get("commands") != expected_commands:
        raise AssertionError("manifest reproduction commands drift")
    for command in expected_commands.values():
        script_tokens = [
            token for token in command.split() if token.startswith("scripts/")
        ]
        if len(script_tokens) == 1 and not (root / script_tokens[0]).is_file():
            raise AssertionError(f"manifest command target is missing: {script_tokens[0]}")
    for relative, expected in manifest["solution_file_sha256"].items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(
                f"champion source drift: {relative}={actual}, expected={expected}"
            )
    return (
        "champion manifest: OK "
        f"({len(manifest['solution_file_sha256'])} source hashes, "
        f"contention-controlled exact row {exact_status})"
    )


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    print(validate_manifest(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
