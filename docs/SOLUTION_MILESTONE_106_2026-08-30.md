# Solution milestone 106 — case-9 frozen half-output promotion

Date: 2026-08-30 SGT

## Outcome

Exact all-valid float32 case 9 now keeps its attention output projection in
internal fp16 before returning immediately to fp32. The new boundary is enabled
only inside the existing `B64/S128/D128/H1/F128/L4/causal` gate with no
effective mask. Packed-half QKV, half FFN and the freeze-only graph remain
unchanged. Padded masks, other dtypes/devices, near shapes, training and graph
failures retain their previous paths.

The interaction was previously untested after packed-half QKV entered the
frozen graph. Five pre-integration sessions and five reverse post-integration
removal sessions were all favorable. Production is retained on causal A/B
evidence even though environmental variance lowered the new three-session
whole-suite mean from the historical 2.501744x point to **2.486131x**. The
cross-session decrease is disclosed and is not attributed to this case-9
change.

## Pre-integration interaction gate

The candidate was the current frozen packed-half case-9 graph plus half output
projection. Each session used seven explicit trials, seven warmups and five
alternating rounds of twenty samples:

```bash
/usr/bin/time -l .venv/bin/python \
  experiments/benchmark_torchscript_solution.py \
  --case 9 --mode production --control-production \
  --incremental-half-output --seed SEED \
  --accuracy-trials 7 --warmup 7 --repeats 20 --rounds 5
```

Seeds 8600–8604 measured `1.053819x`, `1.060529x`, `1.064222x`,
`1.062631x` and `1.064329x`, an all-positive **1.061099x geometric
mean**. All 35 explicit trials passed; the largest maximum absolute error was
`0.00177007914`. Sessions took 2.04–2.08 seconds real, used at most
304,201,728 bytes RSS and reported at most 429,933,504 bytes peak footprint.

## Reverse post-integration gate

After the narrow production flag landed, the candidate leg removed only that
flag. The harness reports `production latency / removal latency`, so a value
below one means production is faster:

```bash
/usr/bin/time -l .venv/bin/python \
  experiments/benchmark_torchscript_solution.py \
  --case 9 --mode production --control-production \
  --case9-remove-half-output --seed SEED \
  --accuracy-trials 7 --warmup 7 --repeats 20 --rounds 5
```

Seeds 8610–8614 reported `0.956820x`, `0.940198x`, `0.938460x`,
`0.950651x` and `0.936819x`. In the intuitive production-over-removal
direction these are `1.045129x`, `1.063606x`, `1.065576x`, `1.051911x`
and `1.067442x`, a **1.058697x geometric mean**. All 35 explicit trials and
all continuity checks passed.

## Broad correctness and current score

A fresh fail-fast matrix at seed 8620 passed **117/117** trials across all 39
case/dtype combinations. Native float16 and bfloat16 remained bit-identical to
the reference. All **65/65** repository tests pass, including a permanent
all-valid case-9 half-output cache assertion.

```bash
for dtype_name in float32 float16 bfloat16; do
  for case_id in {1..13}; do
    PYTHONPATH=. .venv/bin/python \
      experiments/benchmark_sdpa_candidate.py \
      --candidate solution --case "$case_id" --device mps \
      --dtype "$dtype_name" --accuracy-trials 3 --seed 8620 --skip-timing
  done
done
```

Three complete synchronized sessions used three accuracy trials per row, two
warmups and four alternating five-sample timing rounds:

```bash
for case_id in {1..13}; do
  PYTHONPATH=. .venv/bin/python \
    experiments/benchmark_sdpa_candidate.py \
    --candidate solution --case "$case_id" --device mps --dtype float32 \
    --accuracy-trials 3 --seed SEED --warmup 2 --repeats 5 --rounds 4
done
```

| Case | Seed 8625 | Seed 8635 | Seed 8645 |
|---:|---:|---:|---:|
| 1 | 2.471089x | 2.458247x | 2.447712x |
| 2 | 2.268921x | 2.202288x | 1.648398x |
| 3 | 1.793481x | 1.912192x | 1.873979x |
| 4 | 2.307432x | 2.293920x | 2.197521x |
| 5 | 2.475696x | 2.519216x | 2.486495x |
| 6 | 2.362820x | 2.364259x | 2.378957x |
| 7 | 3.189063x | 3.121846x | 3.101383x |
| 8 | 2.953583x | 3.059039x | 2.813100x |
| 9 | 2.024394x | 1.989661x | 1.975903x |
| 10 | 2.081766x | 2.061434x | 2.142718x |
| 11 | 3.322970x | 3.392840x | 3.346483x |
| 12 | 1.854682x | 1.906038x | 1.879031x |
| 13 | 4.472902x | 4.558084x | 4.794203x |
| **Geometric mean** | **2.499365x** | **2.517673x** | **2.441356x** |

The arithmetic mean of the three session points is **2.486131x** and their
geometric mean is `2.485918x`. Every row in every session passed correctness
and remained faster than its same-session baseline. Case 9's three new points
have a 1.996549x geometric mean versus 1.879967x for the preceding champion's
three points, but that cross-session 1.062013x ratio is supporting context; the
ten direct paired sessions above are the causal promotion evidence.

Case 14 remains isolated. A fresh S8192 run loaded the actual production safe
BK48 kernel and passed all 8,388,608 values against both BK32 continuity
(`max_abs=0.000918507576`) and the explicit reference
(`max_abs=0.000949084759`). It took 12.93 seconds real, used 977,567,744 bytes
RSS and reported 15,765,505,176 bytes peak footprint including the explicit
reference. The exact batch-32 observation remains 97.173 seconds; no full
target reference, official aggregate or MFU is inferred.

```bash
/usr/bin/time -l env \
  TORCH_EXTENSIONS_DIR="$PWD/artifacts/m106-case14-boundary-extensions" \
  PYTHONPATH=. .venv/bin/python \
  experiments/benchmark_case14_nax_qprescale_bk64.py \
  --batch-size 1 --seq-len 8192 --seed 8650 --candidate-bk 48 \
  --production-candidate --explicit-reference --skip-timing
```

No submission, push, publication, organizer contact or repository-visibility
action occurred.
