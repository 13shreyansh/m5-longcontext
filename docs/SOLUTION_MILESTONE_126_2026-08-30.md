# Milestone 126 — organizer-default matmul-precision alignment

Timestamp: **2026-08-30 08:16 SGT**

## Contract gap

The official PyTorch harness calls
`torch.set_float32_matmul_precision(args.matmul_precision)` and defaults to
`high`. The synchronized experiment harness and exact case-14 runner had not
set the value, so PyTorch's process default remained `highest`. PyTorch 2.8
documents the performance effect as CUDA-facing, but the installed MPS runtime
was measured rather than assumed.

Two experiment-only flags now expose the setting and print its effective value:

- `experiments/benchmark_sdpa_candidate.py --matmul-precision ...`
- `experiments/run_case14_solution.py --matmul-precision ...`

No production source or dispatch gate changed.

## Three-setting screen

Fresh processes screened cases 3, 6 and 13 at `highest`, official-default
`high`, and `medium`:

```bash
for case_id in 3 6 13; do
  for precision_name in highest high medium; do
    /usr/bin/time -l env PYTHONPATH=. .venv/bin/python \
      experiments/benchmark_sdpa_candidate.py \
      --candidate solution --case "$case_id" --device mps --dtype float32 \
      --matmul-precision "$precision_name" --accuracy-trials 3 \
      --seed $((8800 + case_id)) --warmup 3 --repeats 5 --rounds 3
  done
done
```

All 27 comparisons passed. Each case reported identical error summaries across
the three settings for its fixed seeds. Timings are cross-process observations,
not paired causal effects:

| Case | Setting | Baseline ms | Solution ms | Speedup |
|---:|---|---:|---:|---:|
| 3 | highest | 1.502458 | 0.784292 | 1.915687x |
| 3 | high | 1.520166 | 0.868250 | 1.750839x |
| 3 | medium | 1.655500 | 0.823708 | 2.009814x |
| 6 | highest | 1173.599125 | 483.060542 | 2.429507x |
| 6 | high | 1246.266500 | 501.641500 | 2.484377x |
| 6 | medium | 1520.915250 | 643.356292 | 2.364033x |
| 13 | highest | 419.792041 | 79.793167 | 5.261002x |
| 13 | high | 447.075667 | 83.340083 | 5.364474x |
| 13 | medium | 453.884042 | 86.814792 | 5.228188x |

`medium` raised absolute solution latency on the large case-6 and long
case-13 probes and offers no defensible run-contract advantage. It is rejected.
The mixed `high`/`highest` ordering is consistent with process and thermal
variance; no causal MPS precision effect is claimed.

## Complete official-default point

One complete synchronized cases-1–13 point explicitly used `high`, three
accuracy trials per row, two warmups, and four alternating five-sample rounds:

```bash
for case_id in {1..13}; do
  /usr/bin/time -l env PYTHONPATH=. .venv/bin/python \
    experiments/benchmark_sdpa_candidate.py \
    --candidate solution --case "$case_id" --device mps --dtype float32 \
    --matmul-precision high --accuracy-trials 3 --seed 8810 \
    --warmup 2 --repeats 5 --rounds 4
done
```

All **39/39** comparisons passed.

| Case | Speedup |
|---:|---:|
| 1 | 2.443338x |
| 2 | 2.177025x |
| 3 | 2.036585x |
| 4 | 2.272273x |
| 5 | 2.426687x |
| 6 | 2.276272x |
| 7 | 3.180121x |
| 8 | 3.233811x |
| 9 | 1.917192x |
| 10 | 2.004113x |
| 11 | 3.299750x |
| 12 | 1.844348x |
| 13 | 4.877130x |
| **Geometric mean** | **2.513080x** |

The point lies inside the preceding process-default range
`2.441356x–2.517673x`. The case-6 command was the longest at 83.48 seconds
real and reported 21,970,273,168 bytes peak footprint. Case 8 reported the
largest host RSS, 533,315,584 bytes. `pmset -g therm` still reported no
recorded thermal or performance warning after the matrix.

## Exact case 14 under the official default

Preflight reported AC power, 100% charge, and no recorded thermal or
performance warning. The unchanged exact production route then ran with the
same seed as the prior repeatability pair:

```bash
/usr/bin/time -l env PYTHONPATH=. .venv/bin/python \
  experiments/run_case14_solution.py \
  --batch-size 32 --seq-len 100000 --dtype float32 --seed 8750 \
  --matmul-precision high
```

The command exited `0`:

```text
shape=(32, 100000, 1024) dtype=torch.float32 finite=True
elapsed_ms=98588.985210 matmul_precision=high
mean=-3.64500883e-12 std=0.999995361 max_abs=13.6145725
mps_current_gib=26.037539 mps_driver_gib=30.489014
process_max_rss_gib=0.286209
```

It took 106.54 seconds real, used 307,314,688 bytes maximum RSS, and reported
33,079,231,448 bytes peak footprint. All 3,276,800,000 outputs were finite.
The output summary and driver allocation match the preceding same-seed runs;
98.589 seconds is within their 69.605–131.088-second range and is close to the
97.173-second historical median.

## Decision and whole-suite scorecard

Use explicit `--matmul-precision high` in all future authoritative commands and
the final run contract. Reject `medium`. Preserve the earlier three
process-default points as historical evidence rather than pooling settings.

| Published rows | Organizer-default evidence | Historical context |
|---|---|---|
| 1–13 | `2.513080x`; 39/39 fresh float32 trials | process-default points `2.499365x`, `2.517673x`, `2.441356x`; 117/117 preceding matrix; 65/65 tests |
| 14 | exact B32/S100000 `98.588985` seconds; all outputs finite | process-default `69.605358`, `97.172911`, `131.088197` seconds; paired BK48/BK32 `1.082579x` |

No official MFU or organizer score is inferred.
