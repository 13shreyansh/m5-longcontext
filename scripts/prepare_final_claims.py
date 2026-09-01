#!/usr/bin/env python3
"""Validate a completed row-14 protocol and emit canonical claim values.

This script is deliberately read-only. It does not edit the champion manifest,
storyboard, Devpost draft, repository state, or any external service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

try:
    from scripts.run_case14_final_protocol import (
        DEFAULT_QUIET_WINDOW_SECONDS,
        DEFAULT_OUTPUT,
        EXPECTED_SHAPE,
        ROOT,
        parse_runner_output,
        runner_command,
        validate_measurement_snapshot,
    )
except ModuleNotFoundError:  # Standalone execution puts scripts/ on sys.path.
    from run_case14_final_protocol import (  # type: ignore[no-redef]
        DEFAULT_QUIET_WINDOW_SECONDS,
        DEFAULT_OUTPUT,
        EXPECTED_SHAPE,
        ROOT,
        parse_runner_output,
        runner_command,
        validate_measurement_snapshot,
    )


MAX_CLAIMABLE_MAX_OVER_MIN = 1.15


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-9)


def validate_completed_summary(summary: dict, logs: list[dict]) -> dict:
    if summary.get("status") != "complete":
        raise AssertionError("final protocol summary is not complete")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise AssertionError("final protocol metadata is missing")
    expected_protocol = {
        "runs": 3,
        "seed": 9200,
        "cooldown_seconds": 300,
        "warmup_items_outside_measured_interval": 1,
        "matmul_precision": "high",
        "trimmed_samples": 0,
        "control_policy": "power_agnostic_external_compute_contention_monitoring",
        "power_state_checked": False,
        "runtime_process_poll_seconds": 1,
        "quiet_window_seconds": DEFAULT_QUIET_WINDOW_SECONDS,
        "quiet_window_before_each_run": True,
    }
    if protocol != expected_protocol:
        raise AssertionError(f"unexpected final protocol: {protocol}")
    if summary.get("official_mfu_inferred") is not False:
        raise AssertionError("summary must explicitly refuse official MFU inference")

    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) != 3 or len(logs) != 3:
        raise AssertionError("final protocol requires exactly three runs and logs")
    elapsed_ms = []
    log_hashes = []
    snapshot_times: list[datetime] = []
    for index, (run, log) in enumerate(zip(runs, logs), start=1):
        if run.get("run") != index:
            raise AssertionError(f"unexpected run index at position {index}")
        if run.get("valid") is not True or run.get("failures") != []:
            raise AssertionError(f"run {index} is not valid")
        if run.get("post_environment_failures") != []:
            raise AssertionError(f"run {index} has a post-run environment failure")
        if run.get("shape") != EXPECTED_SHAPE:
            raise AssertionError(f"run {index} has an unexpected shape")
        if run.get("finite") is not True:
            raise AssertionError(f"run {index} output is not entirely finite")
        if run.get("matmul_precision") != "high":
            raise AssertionError(f"run {index} precision is not high")
        if run.get("item_count") != 32:
            raise AssertionError(f"run {index} does not contain 32 items")

        expected_command = runner_command(9200)
        if log.get("command") != expected_command:
            raise AssertionError(f"run {index} command drift")
        if log.get("returncode") != 0:
            raise AssertionError(f"run {index} log return code is nonzero")
        if log.get("runtime_competing_process_failures") != []:
            raise AssertionError(
                f"run {index} lacks a clean runtime competing-process record"
            )
        for boundary in ("before", "after"):
            snapshot = log.get(boundary)
            if not isinstance(snapshot, dict):
                raise AssertionError(f"run {index} {boundary} snapshot is missing")
            timestamp = snapshot.get("timestamp_local")
            if not isinstance(timestamp, str):
                raise AssertionError(
                    f"run {index} {boundary} snapshot timestamp is missing"
                )
            try:
                parsed_timestamp = datetime.fromisoformat(timestamp)
            except ValueError as exc:
                raise AssertionError(
                    f"run {index} {boundary} snapshot timestamp is invalid"
                ) from exc
            if parsed_timestamp.tzinfo is None:
                raise AssertionError(
                    f"run {index} {boundary} snapshot timestamp has no timezone"
                )
            snapshot_times.append(parsed_timestamp)
            failures = validate_measurement_snapshot(snapshot)
            if failures:
                raise AssertionError(
                    f"run {index} {boundary} environment failure: "
                    f"{'; '.join(failures)}"
                )

        parsed_log = parse_runner_output(str(log.get("stdout", "")), 0)
        if parsed_log.get("valid") is not True:
            raise AssertionError(f"run {index} raw stdout is invalid")
        for field in (
            "shape",
            "finite",
            "matmul_precision",
            "item_count",
            "elapsed_ms",
            "item_min_ms",
            "item_median_ms",
            "item_max_ms",
            "first_quarter_median_ms",
            "last_quarter_median_ms",
            "last_over_first",
            "linear_slope_ms_per_item",
        ):
            if parsed_log[field] != run[field]:
                raise AssertionError(f"run {index} summary/log drift: {field}")

        observed_elapsed = float(run["elapsed_ms"])
        if observed_elapsed <= 0:
            raise AssertionError(f"run {index} elapsed time is not positive")
        elapsed_ms.append(observed_elapsed)
        encoded = json.dumps(log, indent=2, sort_keys=True).encode() + b"\n"
        log_hashes.append(hashlib.sha256(encoded).hexdigest())

    if snapshot_times != sorted(snapshot_times):
        raise AssertionError("final protocol snapshot timestamps are not monotonic")

    declared_elapsed = summary.get("elapsed_ms")
    if declared_elapsed != elapsed_ms:
        raise AssertionError("summary elapsed list drift")
    expected_statistics = {
        "median_elapsed_ms": statistics.median(elapsed_ms),
        "min_elapsed_ms": min(elapsed_ms),
        "max_elapsed_ms": max(elapsed_ms),
        "max_over_min": max(elapsed_ms) / min(elapsed_ms),
    }
    for field, expected in expected_statistics.items():
        if not _close(float(summary.get(field, math.nan)), expected):
            raise AssertionError(f"summary statistic drift: {field}")
    if expected_statistics["max_over_min"] > MAX_CLAIMABLE_MAX_OVER_MIN:
        raise AssertionError(
            "final protocol timing spread is not claimable: "
            f"max_over_min={expected_statistics['max_over_min']:.6f} exceeds "
            f"local evidence gate {MAX_CLAIMABLE_MAX_OVER_MIN:.6f}"
        )

    return {
        "schema_version": 1,
        "status": "validated_final_claim_values",
        "row_14_current_route_exact_b32_status": (
            "complete_contention_controlled_protocol"
        ),
        "row_14_current_route_exact_b32_seconds": [value / 1000 for value in elapsed_ms],
        "row_14_current_route_exact_b32_median_seconds": statistics.median(elapsed_ms)
        / 1000,
        "row_14_current_route_exact_b32_min_seconds": min(elapsed_ms) / 1000,
        "row_14_current_route_exact_b32_max_seconds": max(elapsed_ms) / 1000,
        "row_14_current_route_exact_b32_max_over_min": max(elapsed_ms)
        / min(elapsed_ms),
        "matmul_precision": "high",
        "runs": 3,
        "seed": 9200,
        "cooldown_seconds": 300,
        "control_policy": "power_agnostic_external_compute_contention_monitoring",
        "power_state_checked": False,
        "runtime_process_poll_seconds": 1,
        "quiet_window_seconds": DEFAULT_QUIET_WINDOW_SECONDS,
        "quiet_window_before_each_run": True,
        "all_outputs_finite": True,
        "protocol_started_at_local": snapshot_times[0].isoformat(),
        "protocol_finished_at_local": snapshot_times[-1].isoformat(),
        "official_mfu_or_combined_score": None,
        "log_sha256": log_hashes,
    }


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def record_from_summary_path(summary_path: Path) -> dict:
    summary_path = summary_path.resolve()
    if not _within(summary_path, DEFAULT_OUTPUT):
        raise ValueError(f"summary must be under ignored {DEFAULT_OUTPUT}")
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    logs = []
    for run in summary.get("runs", []):
        log_path = (ROOT / str(run.get("log", ""))).resolve()
        if not _within(log_path, DEFAULT_OUTPUT):
            raise ValueError(f"run log must be under ignored {DEFAULT_OUTPUT}: {log_path}")
        logs.append(json.loads(log_path.read_text()))
    record = validate_completed_summary(summary, logs)
    record["summary_sha256"] = hashlib.sha256(summary_bytes).hexdigest()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    record = record_from_summary_path(args.summary)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
