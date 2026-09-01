from scripts.run_case14_final_protocol import (
    DEFAULT_QUIET_WINDOW_SECONDS,
    EXPECTED_SHAPE,
    contention_only,
    contender_identity,
    external_compute_contenders,
    parse_runner_output,
    privacy_safe_snapshot,
    summarize_contenders,
    validate_measurement_snapshot,
    validate_process_snapshot,
    timing_spread_exceeds,
)


def snapshot():
    def result(stdout: str):
        return {"returncode": 0, "stdout": stdout, "stderr": "", "command": []}

    return {
        "timestamp_local": "2026-08-31T12:00:00+08:00",
        "memory_pressure": result("System-wide memory free percentage: 20%\n"),
        "swap": result("vm.swapusage: total = 0.00M  used = 0.00M\n"),
        "processes": result("0.0 /usr/bin/true\n"),
    }


def test_accepts_power_agnostic_snapshot_and_rejects_external_compute_contention():
    assert DEFAULT_QUIET_WINDOW_SECONDS == 60
    state = snapshot()
    assert validate_measurement_snapshot(state) == []
    assert set(state) == {"timestamp_local", "memory_pressure", "swap", "processes"}
    competing_state = snapshot()
    competing_state["processes"]["stdout"] = (
        "88.0 /other/work/.venv/bin/python long_job.py\n"
    )
    assert "competing high-CPU compute process detected" in (
        validate_measurement_snapshot(competing_state)
    )
    assert validate_process_snapshot(competing_state["processes"]) == [
        "competing high-CPU compute process detected"
    ]
    assert external_compute_contenders(competing_state["processes"]) == [
        {
            "cpu_percent": 88.0,
            "command": "/other/work/.venv/bin/python long_job.py",
        }
    ]
    assert contention_only(validate_measurement_snapshot(competing_state)) is True
    assert contention_only(["process query failed"]) is False
    assert contention_only([]) is False

    node_state = snapshot()
    node_state["processes"]["stdout"] = (
        "125.0 /other/work/node apps/server/dist/index.js\n"
    )
    assert external_compute_contenders(node_state["processes"]) == [
        {
            "cpu_percent": 125.0,
            "command": "/other/work/node apps/server/dist/index.js",
        }
    ]

    codex_job_state = snapshot()
    codex_job_state["processes"]["stdout"] = (
        "75.0 /usr/local/bin/codex exec --full-auto optimize-other-track\n"
    )
    assert external_compute_contenders(codex_job_state["processes"]) == [
        {
            "cpu_percent": 75.0,
            "command": (
                "/usr/local/bin/codex exec --full-auto optimize-other-track"
            ),
        }
    ]


def test_ignores_codex_control_plane_processes():
    state = snapshot()
    state["processes"]["stdout"] = (
        "91.7 /Applications/ChatGPT.app/Contents/Resources/codex "
        "-c features.code_mode_host=true app-server\n"
        "160.8 /Applications/ChatGPT.app/Contents/Frameworks/"
        "Codex Framework.framework/Helpers/Codex (Renderer) --type=renderer\n"
    )
    assert external_compute_contenders(state["processes"]) == []
    assert validate_process_snapshot(state["processes"]) == []


def test_summarizes_contenders_without_command_arguments():
    contenders = [
        {
            "cpu_percent": 88.25,
            "command": "/other/work/.venv/bin/python private_job.py --token hidden",
        },
        {
            "cpu_percent": 125.0,
            "command": "/other/work/node secret-script.js --password hidden",
        },
    ]
    assert summarize_contenders(contenders) == "node:125.0,python:88.2"
    assert "private_job" not in summarize_contenders(contenders)
    assert "secret" not in summarize_contenders(contenders)
    changed_cpu = [dict(contenders[0], cpu_percent=51.0), contenders[1]]
    assert contender_identity(contenders) == "node,python"
    assert contender_identity(changed_cpu) == contender_identity(contenders)

    state = snapshot()
    state["processes"]["stdout"] = (
        "88.2 /private/work/.venv/bin/python private_job.py --token hidden\n"
    )
    redacted = privacy_safe_snapshot(state)
    assert redacted["processes"] == {
        "returncode": 0,
        "observed_process_count": 1,
        "external_compute_contender_count": 1,
        "external_compute_contenders": "python:88.2",
        "stderr_present": False,
        "command_lines_redacted": True,
    }
    rendered = str(redacted)
    assert "/private/work" not in rendered
    assert "private_job" not in rendered
    assert "hidden" not in rendered


def test_rejects_irrecoverable_partial_timing_spread():
    assert timing_spread_exceeds([61791.056293]) is False
    assert timing_spread_exceeds([61791.056293, 84202.070290]) is True
    assert timing_spread_exceeds([67786.083418, 67797.913580]) is False


def test_rejects_missing_or_failed_observability_queries():
    state = snapshot()
    state.pop("memory_pressure")
    state["swap"]["returncode"] = 1
    assert validate_measurement_snapshot(state) == [
        "memory_pressure snapshot is missing",
        "swap query failed",
    ]


def test_parses_complete_exact_run():
    output = (
        f"shape={EXPECTED_SHAPE} dtype=torch.float32 finite=True "
        "elapsed_ms=98588.985000 matmul_precision=high\n"
        "item_count=32 item_min_ms=1000.000000 item_median_ms=2000.000000 "
        "item_max_ms=5000.000000 first_quarter_median_ms=1500.000000 "
        "last_quarter_median_ms=2500.000000 last_over_first=1.666667 "
        "linear_slope_ms_per_item=12.500000\n"
    )
    parsed = parse_runner_output(output, 0)
    assert parsed["valid"] is True
    assert parsed["item_count"] == 32
    assert parsed["elapsed_ms"] == 98588.985


def test_rejects_nonfinite_or_wrong_precision_run():
    output = (
        f"shape={EXPECTED_SHAPE} dtype=torch.float32 finite=False "
        "elapsed_ms=1.000000 matmul_precision=medium\n"
        "item_count=31 item_min_ms=1.000000 item_median_ms=1.000000 "
        "item_max_ms=1.000000 first_quarter_median_ms=1.000000 "
        "last_quarter_median_ms=1.000000 last_over_first=1.000000 "
        "linear_slope_ms_per_item=0.000000\n"
    )
    parsed = parse_runner_output(output, 0)
    assert parsed["valid"] is False
    assert "output is not entirely finite" in parsed["failures"]
    assert "unexpected matmul precision medium" in parsed["failures"]
    assert "unexpected item count 31" in parsed["failures"]
