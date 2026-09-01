from copy import deepcopy

import pytest

from scripts.prepare_final_claims import validate_completed_summary
from scripts.run_case14_final_protocol import EXPECTED_SHAPE, runner_command


def measurement_snapshot(timestamp: str):
    def result(stdout: str):
        return {"returncode": 0, "stdout": stdout, "stderr": "", "command": []}

    return {
        "timestamp_local": timestamp,
        "memory_pressure": result("System-wide memory free percentage: 20%\n"),
        "swap": result("vm.swapusage: total = 0.00M  used = 0.00M\n"),
        "processes": result("0.0 /usr/bin/true\n"),
    }


def complete_fixture():
    elapsed = [98000.0, 99000.0, 100000.0]
    runs = []
    logs = []
    for index, value in enumerate(elapsed, start=1):
        stdout = (
            f"shape={EXPECTED_SHAPE} dtype=torch.float32 finite=True "
            f"elapsed_ms={value:.6f} matmul_precision=high\n"
            "item_count=32 item_min_ms=1000.000000 item_median_ms=2000.000000 "
            "item_max_ms=5000.000000 first_quarter_median_ms=1500.000000 "
            "last_quarter_median_ms=2500.000000 last_over_first=1.666667 "
            "linear_slope_ms_per_item=12.500000\n"
        )
        run = {
            "run": index,
            "log": f"artifacts/final-measurement/fake/run_{index}.json",
            "valid": True,
            "failures": [],
            "post_environment_failures": [],
            "shape": EXPECTED_SHAPE,
            "finite": True,
            "elapsed_ms": value,
            "matmul_precision": "high",
            "item_count": 32,
            "item_min_ms": 1000.0,
            "item_median_ms": 2000.0,
            "item_max_ms": 5000.0,
            "first_quarter_median_ms": 1500.0,
            "last_quarter_median_ms": 2500.0,
            "last_over_first": 1.666667,
            "linear_slope_ms_per_item": 12.5,
        }
        runs.append(run)
        logs.append(
            {
                "command": runner_command(9200),
                "before": measurement_snapshot(
                    f"2026-08-30T22:{index * 10:02d}:00+08:00"
                ),
                "returncode": 0,
                "stdout": stdout,
                "stderr": "",
                "after": measurement_snapshot(
                    f"2026-08-30T22:{index * 10 + 1:02d}:00+08:00"
                ),
                "runtime_competing_process_failures": [],
            }
        )
    summary = {
        "status": "complete",
        "protocol": {
            "runs": 3,
            "seed": 9200,
            "cooldown_seconds": 300,
            "warmup_items_outside_measured_interval": 1,
            "matmul_precision": "high",
            "trimmed_samples": 0,
            "control_policy": (
                "power_agnostic_external_compute_contention_monitoring"
            ),
            "power_state_checked": False,
            "runtime_process_poll_seconds": 1,
            "quiet_window_seconds": 60,
            "quiet_window_before_each_run": True,
        },
        "runs": runs,
        "elapsed_ms": elapsed,
        "median_elapsed_ms": 99000.0,
        "min_elapsed_ms": 98000.0,
        "max_elapsed_ms": 100000.0,
        "max_over_min": 100000.0 / 98000.0,
        "official_mfu_inferred": False,
    }
    return summary, logs


def test_accepts_complete_three_run_summary():
    summary, logs = complete_fixture()
    record = validate_completed_summary(summary, logs)
    assert record["row_14_current_route_exact_b32_seconds"] == [98.0, 99.0, 100.0]
    assert record["row_14_current_route_exact_b32_median_seconds"] == 99.0
    assert record["row_14_current_route_exact_b32_status"] == (
        "complete_contention_controlled_protocol"
    )
    assert record["power_state_checked"] is False
    assert record["official_mfu_or_combined_score"] is None
    assert record["protocol_started_at_local"] == "2026-08-30T22:10:00+08:00"
    assert record["protocol_finished_at_local"] == "2026-08-30T22:31:00+08:00"
    assert len(record["log_sha256"]) == 3


def test_rejects_invalid_run():
    summary, logs = complete_fixture()
    summary["runs"][1]["valid"] = False
    with pytest.raises(AssertionError, match="run 2 is not valid"):
        validate_completed_summary(summary, logs)


def test_rejects_post_run_environment_failure():
    summary, logs = complete_fixture()
    summary["runs"][2]["post_environment_failures"] = ["swap query failed"]
    with pytest.raises(AssertionError, match="post-run environment failure"):
        validate_completed_summary(summary, logs)


def test_rejects_statistic_drift_or_unstable_spread():
    summary, logs = complete_fixture()
    summary["median_elapsed_ms"] = 1.0
    with pytest.raises(AssertionError, match="summary statistic drift"):
        validate_completed_summary(summary, logs)

    summary, logs = complete_fixture()
    unstable = [100000.0, 116000.0, 101000.0]
    for run, log, old, new in zip(summary["runs"], logs, summary["elapsed_ms"], unstable):
        run["elapsed_ms"] = new
        log["stdout"] = log["stdout"].replace(
            f"elapsed_ms={old:.6f}", f"elapsed_ms={new:.6f}"
        )
    summary.update(
        {
            "elapsed_ms": unstable,
            "median_elapsed_ms": 101000.0,
            "min_elapsed_ms": 100000.0,
            "max_elapsed_ms": 116000.0,
            "max_over_min": 1.16,
        }
    )
    with pytest.raises(AssertionError, match="timing spread is not claimable"):
        validate_completed_summary(summary, logs)


def test_rejects_official_mfu_inference():
    summary, logs = complete_fixture()
    summary["official_mfu_inferred"] = True
    with pytest.raises(AssertionError, match="refuse official MFU"):
        validate_completed_summary(summary, logs)


def test_rejects_nonmonotonic_snapshot_timestamps():
    summary, logs = complete_fixture()
    logs[1]["before"]["timestamp_local"] = "2026-08-30T22:00:00+08:00"
    with pytest.raises(AssertionError, match="timestamps are not monotonic"):
        validate_completed_summary(summary, logs)
