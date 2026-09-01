import copy
import json

import pytest

from scripts.verify_champion_manifest import MANIFEST_PATH, main, validate_manifest


def test_champion_manifest_matches_source():
    assert main() == 0
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["verification"]["pytest_passed"] == 111
    assert "release_manifest" in manifest["commands"]
    assert "official_hashes" not in manifest["commands"]


def complete_manifest():
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest["status"] = "final_contention_controlled_exact_measurement_complete"
    manifest["row_14"].update(
        {
            "current_route_exact_b32_status": (
                "complete_contention_controlled_protocol"
            ),
            "current_route_exact_b32_seconds": [98.0, 99.0, 100.0],
            "current_route_exact_b32_median_seconds": 99.0,
            "current_route_exact_b32_min_seconds": 98.0,
            "current_route_exact_b32_max_seconds": 100.0,
            "current_route_exact_b32_max_over_min": 100.0 / 98.0,
        }
    )
    return manifest


def test_accepts_complete_three_run_manifest():
    result = validate_manifest(complete_manifest())
    assert "complete_contention_controlled_protocol" in result


def test_rejects_complete_manifest_statistic_drift():
    manifest = complete_manifest()
    manifest["row_14"]["current_route_exact_b32_median_seconds"] = 1.0
    with pytest.raises(AssertionError, match="statistic drift"):
        validate_manifest(manifest)


def test_rejects_status_mismatch():
    manifest = copy.deepcopy(complete_manifest())
    manifest["status"] = (
        "provisional_pending_contention_controlled_exact_measurement"
    )
    with pytest.raises(AssertionError, match="does not match"):
        validate_manifest(manifest)
