# Milestone 149 — promoted champion clean-export audit

> **Historical checkpoint — superseded for current status.** This file
> preserves the evidence and decisions available at its timestamp. Its
> pending/provisional, battery/power and then-current test-count wording is not
> the submission status. Use the current `README.md`, `TECHNICAL_REPORT.md` and
> `docs/CHAMPION_MANIFEST.json` for current claims.

Timestamp: **2026-08-30 14:51 SGT**

## Outcome

A clean `git archive` of promoted champion commit `622c47c` independently
reacquired both pinned upstreams, verified all organizer hashes, regenerated or
byte-matched every retained source/licence boundary, passed all 72 tests,
compiled the new BM32/BN512 direct-head QKV route from an empty native cache,
passed its S8192 organizer reference, and passed one fresh explicit
organizer-default `high` trial for every ordinary row 1–13.

The first ordinary-row loop accidentally inherited the harness default
`highest`. All 13 passed, but that loop is not treated as the organizer-default
clean gate. The entire loop was repeated with `--matmul-precision high`; only
that successful rerun supports the declared result.

## Exact clean archive command

```bash
set -eu
track_repo="$PWD"
track_python="$track_repo/.venv/bin/python"
audit_dir=$(mktemp -d "$track_repo/artifacts/clean-622c47c-XXXXXX")
git archive 622c47c | tar -x -C "$audit_dir"
cd "$audit_dir"
./scripts/acquire_solution_upstreams.sh
./scripts/verify_official_artifacts.sh
"$track_python" scripts/verify_solution_provenance.py
PYTHONPATH="$audit_dir" "$track_python" -m pytest -q
TORCH_EXTENSIONS_DIR="$audit_dir/.torch-extensions" \
  PYTHONPATH="$audit_dir" "$track_python" \
  experiments/benchmark_case14_nax_qkv_head_layout.py \
  --seq-len 8192 --seed 9260 --skip-timing
for case_id in {1..13}; do
  PYTHONPATH="$audit_dir" "$track_python" \
    experiments/benchmark_sdpa_candidate.py \
    --candidate solution --case "$case_id" --device mps --dtype float32 \
    --accuracy-trials 1 --seed 9260 --skip-timing
done
printf 'CLEAN_DIR=%s\nCLEAN_EXPORT_VERIFICATION=PASS\n' "$audit_dir"
```

The command exited zero and printed `CLEAN_EXPORT_VERIFICATION=PASS` with
`artifacts/clean-622c47c-P7bwFH` as its ignored clean directory. It acquired
exact triton-msl commit `182c1820fd24a836d565e1da842f28414de64084` and MLX
commit `3f0bd54ff0c0af5b88530191d5df31010ce54fcd`. Pytest reported
`72 passed in 25.55s`. The clean native QKV route reported
`tile=(32,512,256,1,8)`, passed the organizer reference with
`0/8,388,608` failures and `max_abs=0.000789582729`, and was bit-identical to
the preceding champion.

## Corrected organizer-default ordinary gate

```bash
set -eu
audit_dir="$PWD/artifacts/clean-622c47c-P7bwFH"
track_python="$PWD/.venv/bin/python"
for case_id in {1..13}; do
  PYTHONPATH="$audit_dir" "$track_python" \
    "$audit_dir/experiments/benchmark_sdpa_candidate.py" \
    --candidate solution --case "$case_id" --device mps --dtype float32 \
    --accuracy-trials 1 --seed 9261 --skip-timing \
    --matmul-precision high
done
printf 'ORGANIZER_DEFAULT_HIGH_CLEAN_ROWS=PASS\n'
```

All 13 commands exited zero and the final marker printed. Every explicit
comparison passed. The largest maximum absolute error was `0.0018384457` on
case 6, followed by `0.0016577244` on case 9; both remain within the organizer
predicate. No timing from this low-power audit is used.

## Resource and environment observations

- Clean archive total: 29 MB.
- Fresh pinned upstreams inside it: 23 MB.
- Fresh native-extension cache: 3.0 MB.
- Battery declined from 24% to 18% while discharging; macOS reported no
  thermal or performance warning.
- The commands were not wrapped in `/usr/bin/time`, so no process maximum RSS
  or peak-footprint claim is made.
- The audit reused the pinned project `.venv`; a new dependency installation
  remains an hours-48–60 release gate.
- The archive, upstream checkouts and extension output remain ignored under
  `artifacts/`; none is committed.

## Whole-suite scorecard

| Published rows | Current evidence | Change |
|---|---|---|
| 1–13 | organizer-default `high` points `2.513080x`/`2.523037x`/`2.508106x`, mean `2.514741x`; 117/117 timed-session trials plus 13/13 clean-export trials | clean committed champion now independently passes every ordinary row |
| 14 | safe BK48 plus promoted BM32/BN512 direct-head QKV; `1.019126x` four-session mean over the preceding projection route; clean S8192 explicit reference passes | clean native source/build/dispatch boundary verified; stable-power exact B32 remains pending |

No official MFU or combined organizer score is inferred. No submission, push,
publication, organizer contact or visibility change occurred.
