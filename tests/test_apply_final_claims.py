import json

import pytest

from scripts.apply_final_claims import (
    CHAMPION,
    DRAFT_JSON,
    DRAFT_MD,
    PUBLIC_README,
    STORYBOARD_JSON,
    STORYBOARD_MD,
    apply_claims,
    atomic_write_texts,
    validate_outputs,
)
from scripts.build_video_assets import render as render_video_assets
from scripts.build_video_captions import render as render_captions
from scripts.build_final_readiness import build_snapshot as build_readiness
from scripts.verify_champion_manifest import validate_manifest
from scripts.verify_devpost_draft import validate_draft
from scripts.verify_video_storyboard import validate_storyboard


def record():
    result = {
        "schema_version": 1,
        "status": "validated_final_claim_values",
        "row_14_current_route_exact_b32_status": (
            "complete_contention_controlled_protocol"
        ),
        "row_14_current_route_exact_b32_seconds": [98.0, 99.0, 100.0],
        "row_14_current_route_exact_b32_median_seconds": 99.0,
        "row_14_current_route_exact_b32_min_seconds": 98.0,
        "row_14_current_route_exact_b32_max_seconds": 100.0,
        "row_14_current_route_exact_b32_max_over_min": 100.0 / 98.0,
        "matmul_precision": "high",
        "runs": 3,
        "seed": 9200,
        "cooldown_seconds": 300,
        "control_policy": "power_agnostic_external_compute_contention_monitoring",
        "power_state_checked": False,
        "runtime_process_poll_seconds": 1,
        "quiet_window_seconds": 60,
        "quiet_window_before_each_run": True,
        "all_outputs_finite": True,
        "protocol_started_at_local": "2026-08-30T22:10:00+08:00",
        "protocol_finished_at_local": "2026-08-30T22:31:00+08:00",
        "official_mfu_or_combined_score": None,
        "log_sha256": ["a" * 64, "b" * 64, "c" * 64],
        "summary_sha256": "d" * 64,
    }
    current = json.loads(CHAMPION.read_text())
    if current.get("status") == "final_contention_controlled_exact_measurement_complete":
        row = current["row_14"]
        protocol = row["current_route_exact_b32_protocol"]
        result.update(
            {
                "row_14_current_route_exact_b32_status": row[
                    "current_route_exact_b32_status"
                ],
                "row_14_current_route_exact_b32_seconds": row[
                    "current_route_exact_b32_seconds"
                ],
                "row_14_current_route_exact_b32_median_seconds": row[
                    "current_route_exact_b32_median_seconds"
                ],
                "row_14_current_route_exact_b32_min_seconds": row[
                    "current_route_exact_b32_min_seconds"
                ],
                "row_14_current_route_exact_b32_max_seconds": row[
                    "current_route_exact_b32_max_seconds"
                ],
                "row_14_current_route_exact_b32_max_over_min": row[
                    "current_route_exact_b32_max_over_min"
                ],
                "matmul_precision": protocol["matmul_precision"],
                "runs": protocol["runs"],
                "seed": protocol["seed"],
                "cooldown_seconds": protocol["cooldown_seconds"],
                "control_policy": protocol["control_policy"],
                "power_state_checked": protocol["power_state_checked"],
                "runtime_process_poll_seconds": protocol[
                    "runtime_process_poll_seconds"
                ],
                "quiet_window_seconds": protocol["quiet_window_seconds"],
                "quiet_window_before_each_run": protocol[
                    "quiet_window_before_each_run"
                ],
                "protocol_started_at_local": protocol["started_at_local"],
                "protocol_finished_at_local": protocol["finished_at_local"],
                "all_outputs_finite": row[
                    "current_route_exact_b32_all_outputs_finite"
                ],
                "log_sha256": row["current_route_exact_b32_log_sha256"],
                "summary_sha256": row[
                    "current_route_exact_b32_summary_sha256"
                ],
            }
        )
    return result


def inputs():
    return (
        json.loads(CHAMPION.read_text()),
        json.loads(DRAFT_JSON.read_text()),
        json.loads(STORYBOARD_JSON.read_text()),
        DRAFT_MD.read_text(),
        STORYBOARD_MD.read_text(),
        PUBLIC_README.read_text(),
    )


def test_applies_one_validated_record_to_every_local_claim_surface():
    expected_median = record()["row_14_current_route_exact_b32_median_seconds"]
    champion, draft, storyboard, draft_md, storyboard_md, public_md = apply_claims(
        record(), *inputs()
    )
    validate_manifest(champion)
    validate_draft(draft, champion, draft_md)
    validate_storyboard(storyboard, champion, storyboard_md)
    assert f"median `{expected_median:.3f}` seconds" in public_md
    assert "PENDING_CONTENTION_CONTROLLED_PROTOCOL" not in draft_md
    assert "PENDING_CONTENTION_CONTROLLED_PROTOCOL" not in storyboard_md
    assert f"MEDIAN {expected_median:.3f} s" in render_video_assets(champion)[
        "02_results.svg"
    ]
    assert f"{expected_median:.3f}-second median" in render_captions(storyboard)
    assert champion["snapshot_sgt"] == record()["protocol_finished_at_local"]
    assets, captions = validate_outputs(
        champion, draft, storyboard, draft_md, storyboard_md
    )
    assert f"MEDIAN {expected_median:.3f} s" in assets["02_results.svg"]
    assert f"{expected_median:.3f}-second median" in captions
    readiness = build_readiness(
        champion,
        draft,
        draft_md,
        storyboard,
        storyboard_md,
        supplied_video_assets=assets,
        supplied_captions=captions,
    )
    assert readiness["local_completion_gates"][
        "final_contention_controlled_row_14_measurement"
    ] is True
    assert "final_contention_controlled_row_14_measurement" not in readiness[
        "blocking_gate_ids"
    ]


def test_rejects_unvalidated_or_official_score_records():
    invalid = record()
    invalid["status"] = "complete"
    with pytest.raises(AssertionError, match="not produced by final validation"):
        apply_claims(invalid, *inputs())
    invalid = record()
    invalid["official_mfu_or_combined_score"] = 72.5
    with pytest.raises(AssertionError, match="invented official score"):
        apply_claims(invalid, *inputs())


def test_second_application_is_idempotent_and_still_rejects_text_drift():
    outputs = apply_claims(record(), *inputs())
    assert apply_claims(record(), *outputs) == outputs
    drifted = list(outputs)
    drifted[3] = drifted[3].replace(
        "**Promoted-route contention-controlled result:**", "**Drifted result:**"
    )
    with pytest.raises(AssertionError, match="pending/completed block counts"):
        apply_claims(record(), *drifted)


def test_atomic_writer_refuses_stale_temporary_file(tmp_path):
    destination = tmp_path / "claim.txt"
    temporary = tmp_path / ".claim.txt.final-claims-tmp"
    temporary.write_text("stale")
    with pytest.raises(FileExistsError, match="stale temporary"):
        atomic_write_texts([(destination, "new")])
    assert temporary.read_text() == "stale"
    assert not destination.exists()
