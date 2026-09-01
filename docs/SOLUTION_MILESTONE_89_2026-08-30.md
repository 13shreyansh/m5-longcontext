# Solution milestone 89 — PyTorch MPS environment controls rejected

Date: 2026-08-30 SGT

## Outcome

PyTorch officially documents `PYTORCH_MPS_FAST_MATH=1` to enable fast math for
MPS kernels and `PYTORCH_MPS_PREFER_METAL=1` to prefer Metal kernels over
MPSGraph for matrix multiplication. Because both are read at process startup,
fresh processes screened default, each switch, and both together.

No mode improved the retained solution. On matmul-heavy case 6, the candidate
medians were effectively identical: default `470.492562` ms, fast math
`470.288271` ms, and prefer-Metal `470.486938` ms. On case 3, default had the
lowest candidate median and the highest baseline-relative speedup. All accuracy
checks passed. No environment variable is added to the run contract, and the
custom case-14 Metal-4 pipeline remains governed by its own explicit safe
compiler policy.

Primary source:
<https://docs.pytorch.org/docs/stable/mps_environment_variables.html>.

## Commands and observed values

Each mode was launched as a fresh process with the variable prefixed to the
same `benchmark_sdpa_candidate.py` command. The case-3 command used three
accuracy trials, two warmups, five repeats and three rounds at seed 8070. The
case-6 command used one accuracy trial, one warmup, two repeats and two rounds
at seed 8071.

| Case | Process environment | Baseline median ms | Candidate median ms | Speedup | Accuracy |
|---:|---|---:|---:|---:|---|
| 3 | default | 1.963667 | 1.045500 | 1.878209x | 3/3 pass |
| 3 | `PYTORCH_MPS_FAST_MATH=1` | 2.241250 | 1.302583 | 1.720620x | 3/3 pass |
| 3 | `PYTORCH_MPS_PREFER_METAL=1` | 1.737292 | 1.065083 | 1.631133x | 3/3 pass |
| 3 | both | 1.733583 | 1.057458 | 1.639387x | 3/3 pass |
| 6 | default | 1138.065021 | 470.492562 | 2.418880x | 1/1 pass |
| 6 | `PYTORCH_MPS_FAST_MATH=1` | 1137.841667 | 470.288271 | 2.419456x | 1/1 pass |
| 6 | `PYTORCH_MPS_PREFER_METAL=1` | 1141.788980 | 470.486938 | 2.426824x | 1/1 pass |

The case-6 candidate spread is only 0.204291 ms, or 0.0434%, and is below a
defensible promotion margin. The case-3 processes were sequential and their
cross-process baseline drift is not treated as a causal latency estimate;
nevertheless, neither candidate absolute median nor within-process speedup
supports a switch. Fast math changed case-6 maximum error by only about
`0.00000003` and all outputs remained valid.

## Whole-suite scorecard after milestone 89

| Published rows | Verified current evidence | Boundary |
|---|---|---|
| 1–13 | Default-environment PyTorch 2.8 champion remains 2.427904x synchronized float32 geometric mean; MPS fast-math/prefer-Metal controls rejected | Cross-process mode screen is diagnostic; no new suite point or environment dependency |
| 14 | Single bounds-safe BK48; five-session BK48-over-BK32 mean 1.082579x; current exact B32 97.173 s | Custom Metal compiler mode was separately gated and retained safe; full explicit target reference infeasible |

Official organizer score and MFU remain **unknown**. No submission, push,
publication, organizer contact or visibility action occurred.
