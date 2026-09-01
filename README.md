# TikTok TechJam 2026 Track 3 — Apple M5 Pro Transformer

This repository contains a device-specific PyTorch/MPS and Metal 4
implementation of the published causal Transformer workload. It targets the
20-core-GPU Apple M5 Pro used for the recorded result. It does not claim an
official organizer score, CUDA portability, or full-length equality with the
physically infeasible explicit row-14 reference.

## Measured result

All timings below are synchronized local measurements on the declared Apple M5
Pro. Compilation and the first launch are outside the measured interval, in
line with the workshop clarification.

| Published rows | Current evidence |
|---|---|
| 1–13 | Three organizer-default `high` sessions measured `2.513080x`, `2.523037x`, and `2.508106x` versus the explicit organizer baseline; arithmetic mean `2.514741x`. All 117 fresh float32 comparisons passed. |
| 14 | The preceding safe route completed the exact `(B=32,S=100000,D=1024,H=16,F=1024,L=2)` execution in `69.605`, `97.173`, and `131.088` seconds, plus an explicit-`high` point at `98.589` seconds; every returned value was finite. The promoted direct-head QKV route measured `1.019126x` over that preceding projection route in four balanced `B=1,S=100000` sessions. Its three contention-controlled exact `B=32` runs were `54.269, 51.896, 52.003` seconds (median `52.003` seconds, range `51.896`–`54.269`), with every output finite. Power state was not inspected or used as a gate. |

The full result, rejected alternatives, and limitations are in
[`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md). The machine-readable final result
identity is in
[`docs/CHAMPION_MANIFEST.json`](docs/CHAMPION_MANIFEST.json).

## Evidence map

The current result has one entry point per verification question:

- implementation, result and limitations:
  [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md);
- machine-readable frozen identity and commands:
  [`docs/CHAMPION_MANIFEST.json`](docs/CHAMPION_MANIFEST.json);
- correctness predicate and coverage:
  [`docs/CORRECTNESS_ORACLE.md`](docs/CORRECTNESS_ORACLE.md);
- timing and aggregation semantics:
  [`docs/PERFORMANCE_MEASUREMENT.md`](docs/PERFORMANCE_MEASUREMENT.md);
- precision and fallback boundaries:
  [`docs/PRECISION_CONTRACT.md`](docs/PRECISION_CONTRACT.md);
- seeds, repeatability and state assumptions:
  [`docs/REPRODUCIBILITY_CONTRACT.md`](docs/REPRODUCIBILITY_CONTRACT.md);
- machine and framework inventory:
  [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md);
- upstream source-to-experiment decisions and retained licences:
  [`docs/SOURCE_TO_EXPERIMENT_LEDGER_2026-08-29.md`](docs/SOURCE_TO_EXPERIMENT_LEDGER_2026-08-29.md),
  [`docs/UPSTREAM_EXPERIMENT_AUDIT_2026-08-29.md`](docs/UPSTREAM_EXPERIMENT_AUDIT_2026-08-29.md),
  and [`docs/DEPENDENCY_LICENSES.md`](docs/DEPENDENCY_LICENSES.md); and
- AI roles, human steering and source disclosure:
  [`docs/AI_TOOL_DISCLOSURE_2026-08-29.md`](docs/AI_TOOL_DISCLOSURE_2026-08-29.md).

Historical milestone files are an audit trail, not the current result identity.
Where an older milestone is linked from the technical report, it supports only
the specific experiment described there. Current claims are controlled by this
README, the technical report and the champion manifest.

## Organizer attachment boundary

The organizer's `torch_transformer_benchmark.py` attachment has no visible
redistribution licence, so it is deliberately absent. Obtain an authorized
copy from the official Track 3 materials and place it locally at:

```text
official/torch_transformer_benchmark.py
```

That local file is ignored and must not be committed. The public-update file
used for the recorded result had SHA-256
`5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`.
See [`official/README.md`](official/README.md) for the exact boundary.

## Environment

The retained champion requires:

- Apple Silicon with MPS and Metal 4 for the accelerated route;
- macOS 26 and Command Line Tools for the runtime Objective-C++ bridge;
- Python with the exact tested resolution in `requirements-lock.txt`; and
- approximately 31 GiB of available unified-memory driver allocation for the
  full row-14 run.

The accelerated NAX route is specific to a compatible Apple GPU. Unsupported
native compilation or dispatch conditions fall back to a bounded PyTorch/Metal
route instead of silently using an unsafe kernel.

## Setup and verification

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-cache-dir --require-hashes \
  --only-binary=:all: -r requirements-lock.txt

# Verifies every distributed file and confirms that no organizer attachment is
# part of the release manifest.
.venv/bin/python scripts/verify_release_manifest.py

# Runs the public-safe tests. The organizer-dependent Transformer module is
# reported as one explicit skip while the authorized attachment is absent.
PYTHONPATH=. .venv/bin/python -m pytest -q

# After placing your authorized organizer attachment under official/:
PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_champion_manifest.py
```

`requirements-solution.txt` records the four direct dependency declarations;
`requirements-lock.txt` freezes all 18 packages in the tested Python 3.9/macOS
arm64 resolution and pins the exact target-wheel SHA-256 for each package.
`--require-hashes --only-binary=:all:` makes installation fail closed if an
artifact differs or the target wheel is unavailable. See
[`docs/DEPENDENCY_LICENSES.md`](docs/DEPENDENCY_LICENSES.md) for dependency and
licence metadata boundaries.

The provenance verifier optionally reacquires the exact ignored upstream
checkouts and byte-checks every generated or vendored source and both retained
MIT licences:

```bash
./scripts/acquire_solution_upstreams.sh
.venv/bin/python scripts/verify_solution_provenance.py
```

## Reproduce the recorded boundaries

One ordinary organizer-reference comparison:

```bash
PYTHONPATH=. .venv/bin/python experiments/benchmark_sdpa_candidate.py \
  --candidate solution --case 1 --device mps --dtype float32 \
  --accuracy-trials 3 --warmup 2 --repeats 5 --rounds 2 \
  --matmul-precision high
```

Safe row-14 preflight only:

```bash
.venv/bin/python scripts/run_case14_final_protocol.py
```

The script deliberately requires the repository-local `.venv`; invoking it
through an unrelated external interpreter is rejected. Preflight output is
privacy-redacted and never prints process arguments or local paths. A
`PRECONDITION_BLOCKED` message and exit code `3` mean the safety check worked
and no benchmark result was produced. Do not treat a blocked preflight as a
failed implementation or a timing observation.

The final protocol deliberately does not inspect battery, charging, thermal,
or power mode. It records memory and swap, rejects observed external high-CPU
Python, Node and Codex contention before, during and after each run, and keeps all three
untrimmed timings subject to a max/min ratio no greater than 1.15. Execution
begins each run only after a 60-second clean monitored-process window and
invalidates the run if any contender appears during measurement. It is
intentionally separate because it takes minutes, allocates tens of GiB, and
writes raw logs under ignored `artifacts/`.
Those raw `--execute` logs can contain local process command lines and paths;
do not commit, paste, screenshot, or publish them without a separate privacy
review.
Execution waits through contention in its initial snapshot, but missing or
failed memory, swap or process observability still blocks immediately. A
partial set whose max/min ratio already exceeds 1.15 stops before another
allocation because no later timing can repair that spread.
After a successful three-run execution, validate the raw logs and emit one
read-only set of canonical claim values with:

```bash
.venv/bin/python scripts/prepare_final_claims.py \
  artifacts/final-measurement/TIMESTAMP/summary.json
```

The claim-preparation command rejects incomplete runs, observed monitored
contention, timing spread above the local evidence gate, summary/log
disagreement, changed protocol settings and any official-MFU inference. It
does not edit the repository or publish anything.

## Correctness boundary

The organizer predicate accepts an element when absolute error is at most
`0.002` **or** relative error is at most `0.02`. Rows 1–13 are checked directly
against the organizer baseline. A full explicit reference for row 14 would
allocate an 18.626-TiB float32 score tensor, so the retained evidence uses an
explicit `S=8192` reference, larger continuity checks, full-length champion
comparisons, and complete-output finiteness checks. That is strong bounded
evidence, but it is not represented as full-length reference equality.

## Source and licence notes

The NAX sources are copied or adapted from Apple MLX commit
`3f0bd54ff0c0af5b88530191d5df31010ce54fcd`. The generated attention and
fused-normalization sources are derived from `triton-msl` commit
`182c1820fd24a836d565e1da842f28414de64084`. Their MIT licences are preserved
under `solution/third_party/`.

Python dependencies are installed rather than vendored. Their exact tested
resolution and installed-wheel licence metadata are recorded in
[`docs/DEPENDENCY_LICENSES.md`](docs/DEPENDENCY_LICENSES.md).

No project-wide reuse licence is granted for the entrant-authored code by this
package. Public visibility for competition review does not itself grant reuse
rights. The bundled third-party files remain governed by their respective
licences.
