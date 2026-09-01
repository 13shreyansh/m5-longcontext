#!/usr/bin/env python3
"""Fail-closed, power-agnostic final measurement for published row 14.

The default mode is preflight-only. Execution requires an explicit --execute.
The protocol never queries or gates on battery, charging, thermal, or power
mode. It records memory/swap state, rejects observed external high-CPU compute
contention, checks every output, and preserves raw logs under ignored
artifacts/.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
RUNNER = ROOT / "experiments" / "run_case14_solution.py"
DEFAULT_OUTPUT = ROOT / "artifacts" / "final-measurement"
EXPECTED_SHAPE = "(32, 100000, 1024)"
DEFAULT_QUIET_WINDOW_SECONDS = 60
MAX_ACCEPTABLE_MAX_OVER_MIN = 1.15
CONTENTION_FAILURE = "competing high-CPU compute process detected"
COMPUTE_PROCESS_PATTERN = re.compile(
    r"\b(?:python(?:[0-9.]*)?|node(?:_repl)?|codex(?:-code-mode-host)?)\b",
    re.IGNORECASE,
)
CODEX_CONTROL_PLANE_PATTERN = re.compile(
    r"(?:/Frameworks/Codex Framework\.framework/|\bcodex\b.*\bapp-server\b)",
    re.IGNORECASE,
)


def capture(command: list[str]) -> dict[str, object]:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def measurement_snapshot() -> dict[str, object]:
    return {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "memory_pressure": capture(["/usr/bin/memory_pressure"]),
        "swap": capture(["/usr/sbin/sysctl", "vm.swapusage"]),
        "processes": capture(["/bin/ps", "-axo", "%cpu=,command="]),
    }


def privacy_safe_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    """Return a stdout-safe view without process commands or local paths."""

    def query_summary(name: str) -> dict[str, object]:
        result = snapshot.get(name)
        if not isinstance(result, dict):
            return {"status": "missing"}
        return {
            "returncode": result.get("returncode"),
            "stdout": str(result.get("stdout", "")),
            "stderr_present": bool(result.get("stderr")),
        }

    processes = snapshot.get("processes")
    contenders = external_compute_contenders(processes)
    if isinstance(processes, dict):
        process_returncode = processes.get("returncode")
        observed_process_count = len(
            [line for line in str(processes.get("stdout", "")).splitlines() if line]
        )
        process_stderr_present = bool(processes.get("stderr"))
    else:
        process_returncode = None
        observed_process_count = 0
        process_stderr_present = False
    return {
        "timestamp_local": snapshot.get("timestamp_local"),
        "memory_pressure": query_summary("memory_pressure"),
        "swap": query_summary("swap"),
        "processes": {
            "returncode": process_returncode,
            "observed_process_count": observed_process_count,
            "external_compute_contender_count": len(contenders),
            "external_compute_contenders": summarize_contenders(contenders),
            "stderr_present": process_stderr_present,
            "command_lines_redacted": True,
        },
    }


def external_compute_contenders(processes: object) -> list[dict[str, object]]:
    if not isinstance(processes, dict):
        return []
    process_text = str(processes.get("stdout", ""))
    contenders = []
    for line in process_text.splitlines():
        match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s+(.+)$", line)
        if match is None:
            continue
        cpu_text, command = match.groups()
        if (
            float(cpu_text) >= 50.0
            and COMPUTE_PROCESS_PATTERN.search(command)
            and not CODEX_CONTROL_PLANE_PATTERN.search(command)
            and str(ROOT) not in command
        ):
            contenders.append(
                {"cpu_percent": float(cpu_text), "command": command}
            )
    return contenders


def validate_process_snapshot(processes: object) -> list[str]:
    failures: list[str] = []
    if not isinstance(processes, dict):
        return ["process snapshot is missing"]
    if processes.get("returncode") != 0:
        failures.append("process query failed")
    if external_compute_contenders(processes):
        failures.append(CONTENTION_FAILURE)
    return failures


def summarize_contenders(contenders: list[dict[str, object]]) -> str:
    """Return privacy-safe executable/CPU labels without command arguments."""
    labels = []
    for contender in contenders:
        command = str(contender.get("command", ""))
        executable = command.split(maxsplit=1)[0] if command else "unknown"
        name = Path(executable).name or "unknown"
        cpu_percent = float(contender.get("cpu_percent", 0.0))
        labels.append(f"{name}:{cpu_percent:.1f}")
    return ",".join(sorted(labels)) or "unknown"


def contender_identity(contenders: list[dict[str, object]]) -> str:
    """Return a stable, case-folded executable identity for log deduplication."""
    names = []
    for contender in contenders:
        command = str(contender.get("command", ""))
        executable = command.split(maxsplit=1)[0] if command else "unknown"
        names.append((Path(executable).name or "unknown").casefold())
    return ",".join(sorted(names)) or "unknown"


def contention_only(failures: list[str]) -> bool:
    return bool(failures) and set(failures) == {CONTENTION_FAILURE}


def timing_spread_exceeds(elapsed_ms: list[float]) -> bool:
    return (
        len(elapsed_ms) >= 2
        and max(elapsed_ms) / min(elapsed_ms) > MAX_ACCEPTABLE_MAX_OVER_MIN
    )


def wait_for_quiet_window(seconds: int) -> None:
    clean_seconds = 0
    last_signature = ""
    while clean_seconds < seconds:
        process_state = capture(["/bin/ps", "-axo", "%cpu=,command="])
        failures = validate_process_snapshot(process_state)
        if failures:
            contenders = external_compute_contenders(process_state)
            identity = contender_identity(contenders)
            if clean_seconds or identity != last_signature:
                print(
                    "quiet_window_reset=external_compute_contention "
                    f"contenders={summarize_contenders(contenders)}",
                    flush=True,
                )
            clean_seconds = 0
            last_signature = identity
        else:
            last_signature = ""
            clean_seconds += 1
            if clean_seconds in {1, seconds}:
                print(
                    f"quiet_window_clean_seconds={clean_seconds}/{seconds}",
                    flush=True,
                )
        if clean_seconds < seconds:
            time.sleep(1)


def validate_measurement_snapshot(snapshot: dict[str, object]) -> list[str]:
    failures: list[str] = []
    for field in ("memory_pressure", "swap"):
        result = snapshot.get(field)
        if not isinstance(result, dict):
            failures.append(f"{field} snapshot is missing")
        elif result.get("returncode") != 0:
            failures.append(f"{field} query failed")
    failures.extend(validate_process_snapshot(snapshot.get("processes")))
    return failures


def parse_runner_output(stdout: str, returncode: int) -> dict[str, object]:
    summary_match = re.search(
        r"shape=(\([^\n]+\)) dtype=torch\.float32 finite=(True|False) "
        r"elapsed_ms=([0-9.]+) matmul_precision=([a-z]+)",
        stdout,
    )
    item_match = re.search(
        r"item_count=(\d+) item_min_ms=([0-9.]+) "
        r"item_median_ms=([0-9.]+) item_max_ms=([0-9.]+) "
        r"first_quarter_median_ms=([0-9.]+) "
        r"last_quarter_median_ms=([0-9.]+) "
        r"last_over_first=([0-9.]+) "
        r"linear_slope_ms_per_item=(-?[0-9.]+)",
        stdout,
    )
    failures = []
    if returncode != 0:
        failures.append(f"runner exited {returncode}")
    if summary_match is None:
        failures.append("runner summary is missing")
    if item_match is None:
        failures.append("32-item summary is missing")

    parsed: dict[str, object] = {"valid": False, "failures": failures}
    if summary_match is not None:
        shape, finite, elapsed_ms, precision = summary_match.groups()
        parsed.update(
            {
                "shape": shape,
                "finite": finite == "True",
                "elapsed_ms": float(elapsed_ms),
                "matmul_precision": precision,
            }
        )
        if shape != EXPECTED_SHAPE:
            failures.append(f"unexpected shape {shape}")
        if finite != "True":
            failures.append("output is not entirely finite")
        if precision != "high":
            failures.append(f"unexpected matmul precision {precision}")
    if item_match is not None:
        (
            item_count,
            item_min,
            item_median,
            item_max,
            first_quarter,
            last_quarter,
            last_over_first,
            slope,
        ) = item_match.groups()
        parsed.update(
            {
                "item_count": int(item_count),
                "item_min_ms": float(item_min),
                "item_median_ms": float(item_median),
                "item_max_ms": float(item_max),
                "first_quarter_median_ms": float(first_quarter),
                "last_quarter_median_ms": float(last_quarter),
                "last_over_first": float(last_over_first),
                "linear_slope_ms_per_item": float(slope),
            }
        )
        if item_count != "32":
            failures.append(f"unexpected item count {item_count}")
    parsed["valid"] = not failures
    return parsed


def runner_command(seed: int) -> list[str]:
    return [
        "/usr/bin/time",
        "-l",
        str(PYTHON),
        str(RUNNER),
        "--batch-size",
        "32",
        "--seq-len",
        "100000",
        "--seed",
        str(seed),
        "--dtype",
        "float32",
        "--matmul-precision",
        "high",
        "--warmup-items",
        "1",
        "--report-item-timings",
    ]


def write_log(
    path: Path,
    command: list[str],
    before: dict[str, object],
    result: subprocess.CompletedProcess[str],
    after: dict[str, object],
    runtime_competing_process_failures: list[dict[str, object]],
) -> None:
    record = {
        "command": command,
        "before": before,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "after": after,
        "runtime_competing_process_failures": runtime_competing_process_failures,
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9200)
    parser.add_argument("--cooldown-seconds", type=int, default=300)
    parser.add_argument(
        "--quiet-window-seconds",
        type=int,
        default=DEFAULT_QUIET_WINDOW_SECONDS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    if args.cooldown_seconds < 0:
        raise ValueError("cooldown-seconds must be non-negative")
    if args.quiet_window_seconds < 0:
        raise ValueError("quiet-window-seconds must be non-negative")
    if not PYTHON.is_file():
        raise RuntimeError(f"missing repository Python: {PYTHON}")

    initial = measurement_snapshot()
    initial_failures = validate_measurement_snapshot(initial)
    print(
        json.dumps(privacy_safe_snapshot(initial), indent=2, sort_keys=True),
        flush=True,
    )
    if initial_failures:
        if args.execute and contention_only(initial_failures):
            print(
                "PRECONDITION_WAIT: initial external compute contention; "
                "entering declared quiet-window monitor",
                flush=True,
            )
        else:
            print(
                "PRECONDITION_BLOCKED: " + "; ".join(initial_failures),
                file=sys.stderr,
                flush=True,
            )
            return 3
    if not args.execute:
        print("PREFLIGHT_PASS: add --execute to run the declared protocol")
        return 0
    run_stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output_dir = args.output_dir / run_stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    parsed_runs = []
    for run_index in range(args.runs):
        if run_index:
            print(
                f"cooldown_seconds={args.cooldown_seconds}",
                flush=True,
            )
            time.sleep(args.cooldown_seconds)
        wait_for_quiet_window(args.quiet_window_seconds)
        before = measurement_snapshot()
        failures = validate_measurement_snapshot(before)
        if failures:
            summary = {
                "status": "blocked",
                "completed_runs": len(parsed_runs),
                "runs": parsed_runs,
                "failures": failures,
                "snapshot": before,
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            print("PRECONDITION_BLOCKED: " + "; ".join(failures), file=sys.stderr)
            return 3

        command = runner_command(args.seed)
        print(f"run={run_index + 1}/{args.runs} command={' '.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        runtime_competing_process_failures = []
        while process.poll() is None:
            process_state = capture(["/bin/ps", "-axo", "%cpu=,command="])
            process_failures = validate_process_snapshot(process_state)
            if process_failures:
                runtime_competing_process_failures.append(
                    {
                        "timestamp_local": datetime.now().astimezone().isoformat(),
                        "failures": process_failures,
                        "contenders": external_compute_contenders(process_state),
                    }
                )
            time.sleep(1)
        stdout, stderr = process.communicate()
        result = subprocess.CompletedProcess(
            command, process.returncode, stdout=stdout, stderr=stderr
        )
        after = measurement_snapshot()
        log_path = output_dir / f"run_{run_index + 1}.json"
        write_log(
            log_path,
            command,
            before,
            result,
            after,
            runtime_competing_process_failures,
        )
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        parsed = parse_runner_output(result.stdout, result.returncode)
        parsed["run"] = run_index + 1
        parsed["log"] = str(log_path.relative_to(ROOT))
        post_environment_failures = validate_measurement_snapshot(after)
        parsed["post_environment_failures"] = post_environment_failures
        if post_environment_failures:
            parsed["failures"].extend(
                f"post-run environment: {failure}"
                for failure in post_environment_failures
            )
            parsed["valid"] = False
        parsed["runtime_competing_process_failures"] = (
            runtime_competing_process_failures
        )
        if runtime_competing_process_failures:
            parsed["failures"].append(
                "competing high-CPU compute process observed during measured run"
            )
            parsed["valid"] = False
        parsed_runs.append(parsed)
        if not parsed["valid"]:
            summary = {"status": "invalid", "runs": parsed_runs}
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            return 2
        partial_elapsed = [float(run["elapsed_ms"]) for run in parsed_runs]
        if timing_spread_exceeds(partial_elapsed):
            spread = max(partial_elapsed) / min(partial_elapsed)
            failure = (
                f"irrecoverable timing spread {spread:.6f} exceeds "
                f"{MAX_ACCEPTABLE_MAX_OVER_MIN:.6f}"
            )
            summary = {
                "status": "invalid",
                "runs": parsed_runs,
                "failures": [failure],
            }
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n"
            )
            print(f"SPREAD_REJECTED: {failure}", file=sys.stderr, flush=True)
            return 2

    elapsed_values = [float(run["elapsed_ms"]) for run in parsed_runs]
    summary = {
        "status": "complete",
        "protocol": {
            "runs": args.runs,
            "seed": args.seed,
            "cooldown_seconds": args.cooldown_seconds,
            "warmup_items_outside_measured_interval": 1,
            "matmul_precision": "high",
            "trimmed_samples": 0,
            "control_policy": (
                "power_agnostic_external_compute_contention_monitoring"
            ),
            "power_state_checked": False,
            "runtime_process_poll_seconds": 1,
            "quiet_window_seconds": args.quiet_window_seconds,
            "quiet_window_before_each_run": True,
        },
        "runs": parsed_runs,
        "elapsed_ms": elapsed_values,
        "median_elapsed_ms": statistics.median(elapsed_values),
        "min_elapsed_ms": min(elapsed_values),
        "max_elapsed_ms": max(elapsed_values),
        "max_over_min": max(elapsed_values) / min(elapsed_values),
        "official_mfu_inferred": False,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary={summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
