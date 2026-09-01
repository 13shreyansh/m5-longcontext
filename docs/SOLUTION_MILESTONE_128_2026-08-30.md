# Milestone 128 — static-graph revalidation under organizer-default `high`

Timestamp: **2026-08-30 08:26 SGT**

## Question

Cases 2, 3, 9 and 12 retain narrow static TorchScript routes, but their direct
promotion comparisons predated the explicit organizer-default
`torch.set_float32_matmul_precision("high")` run contract. A complete `high`
score point was positive, yet it did not isolate whether each graph still beat
its otherwise identical eager route.

The experiment-only TorchScript harness now accepts `--matmul-precision` and
prints the effective value. Production source and dispatch are unchanged.

## Exact command

Each case used seven organizer comparisons, seven warmups, and five alternating
rounds of twenty samples:

```bash
for case_id in 2 3 9 12; do
  /usr/bin/time -l env PYTHONPATH=. .venv/bin/python \
    experiments/benchmark_torchscript_solution.py \
    --case "$case_id" --mode production --matmul-precision high \
    --seed $((8820 + case_id)) --accuracy-trials 7 \
    --warmup 7 --repeats 20 --rounds 5
done
```

Every command exited `0`.

## Results

| Case | Explicit trials | Largest max abs | Continuity | Eager median ms | Production median ms | Speedup | Build/first ms |
|---:|---:|---:|---|---:|---:|---:|---:|
| 2 | 7/7 | 0.00126622617 | pass; bit-identical | 1.527854 | 1.448896 | 1.054495x | 351.118125 |
| 3 | 7/7 | 0.00100445747 | pass; max abs 0.000825747848 | 0.983645 | 0.819021 | 1.201002x | 421.140500 |
| 9 | 7/7 | 0.00169110298 | pass; bit-identical | 2.587167 | 2.476771 | 1.044573x | 363.067167 |
| 12 | 7/7 | 0.00112941861 | pass; bit-identical | 1.082291 | 0.923312 | 1.172183x | 363.982625 |

The commands used at most 293,715,968 bytes RSS and 425,264,040 bytes peak
footprint. Real times were 1.63–1.94 seconds. Compilation and first execution
remain excluded from the measured steady-state window, matching the workshop
clarification.

## Decision and whole-suite scorecard

Retain all four exact gates. Every graph remains faster than its current eager
route under organizer-default `high`; no precision-dependent reversal was
found. Case 9 still has the smallest margin and keeps its already documented
variance caveat.

| Published rows | Current evidence | Change |
|---|---|---|
| 1–13 | official-default point `2.513080x`; fresh 39/39 float32 point; static routes 28/28 explicit plus 4/4 continuity checks under `high`; 65/65 tests | production unchanged; precision-contract confidence increased |
| 14 | official-default exact `98.588985` seconds; all outputs finite; paired BK48/BK32 `1.082579x` | unchanged |

No official MFU or organizer score is inferred.
