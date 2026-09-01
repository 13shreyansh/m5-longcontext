# Track 3 technical report — Apple M5 Pro result

Snapshot: **2026-08-31 SGT**
Status: local challenge-window result; **not submitted and not an official score**

## Executive result

This entry targets the participant's own 20-core-GPU Apple M5 Pro through
PyTorch MPS and Metal. The implementation preserves the organizer's Transformer
parameter names and strict weight-copy contract, leaves both official
attachments unchanged, and selects a measured route by shape and dtype. On the
13 published reference-feasible float32 cases, three complete synchronized
same-machine sessions under the organizer-default float32 matmul setting
`high` measured
**2.513080x, 2.523037x and 2.508106x**, an arithmetic mean of **2.514741x**
(geometric mean `2.514733x`) with only `0.595334%` max/min spread. All 117
fresh float32 comparisons passed. Earlier process-default sessions remain in
the experiment record but are not pooled with these declared points. Exact all-valid cases
2/9/12 use lazy frozen TorchScript graphs, while case 3 adds inference
optimization. Case 2 was promoted after five positive
production-aware comparisons at a **1.098356x** geometric mean and bit-identical
output; cases 9/12 have static-graph production means of **1.028507x** and
**1.151773x**;
case 9 additionally has a ten-session all-positive packed-half-QKV interaction,
with 1.123369x pre-integration and 1.105729x production means;
its later post-packed half-output interaction has five all-positive candidate
sessions at a 1.061099x mean and five all-positive reverse removal sessions at
a 1.058697x production advantage;
case 3 has a five-session promotion mean of **1.230595x** and a fresh 1.238087x
post-refactor check;
compilation and first execution are outside timed runs as permitted by the
workshop. The post-promotion
correctness matrix passed **117/117 trials across 39/39 case/dtype
combinations** (cases 1–13 across float32, float16, and bfloat16) with zero
failed accuracy runs for fresh seed `8620`. The exact width-1024 case 8 narrows
its dense projections and SDPA inputs internally to fp16, while retaining fp32
residuals, normalization, GELU and output; 51/51 all-valid/padded case-8 trials
passed through the latest suite. Exact cases 6 and 13 similarly
narrow QKV, SDPA inputs and attention-output projection while retaining fp32
FFN/residual/norm/output; 23/23 case-6 and 18/18 case-13 trials passed. Six
exact short rows (cases 1, 4, 5, 7, 10 and 11) use that narrower boundary after
91/91 explicit trials over 82,575,360 output values and repeatable positive
same-session timing. The remaining exact cases 2, 3, 9 and 12 use separately
validated full-half, QKV-only, FFN-only and QKV-only boundaries; their 64/64
explicit trials and all 20 paired timing sessions passed. Exact all-valid case
6 additionally uses a generated 128-thread fused residual-plus-LayerNorm
kernel; its integrated four-session mean is 1.032323x including one negative
odd-round observation. Exact all-valid cases 10 and 13 use the corresponding
generated 64-thread kernel after five positive sessions each, with geometric
means 1.018419x and 1.018194x. Exact case 7 uses the generated 32-thread
kernel after five positive sessions with a 1.073999x mean.

The published stress row `(B=32,S=100000,D=1024,H=16,F=1024,L=2)` cannot run
through the explicit reference because one float32 attention-score tensor would
require **18.626 TiB**. The current stress route promotes a direct-head fp16
packed-QKV NAX projection plus fp16 attention-output and FFN projections inside
the bounded float32 path while returning every linear result immediately to
fp32. The QKV projection uses `BM32/BN512/BK256/WM1/WN8` and writes the 48
head-major tensors directly. Four balanced S100000 sessions were champion-bit-
identical and measured `1.011633x`/`1.021096x`/`1.021071x`/`1.022742x`, a
**1.019126x** mean over the preceding packed-half projection plus layout route,
with 33/36 positive pairs. The unmodified production runner also completed
B1/S100000 under explicit `high` with every output finite in 3036.870 ms; that
batch-one result is a dispatch check rather than the final exact batch-32
measurement. Its main attention kernel is
Apple's pinned MLX NAX template, locally instantiated as
`BQ256/BK48/BD64/WM16/WN1`, invoked directly on PyTorch MPS storage through a
narrow Metal 4 bridge. Q is padded by 96 rows to enable the aligned BQ256
specialization while K/V remain at 100,000; padded outputs are discarded. The
kernel retains the four query fragments and applies the log-base-adjusted scale
once to Q. BK48's odd three-fragment score width is stored in a rounded four-
fragment tile while max, exponential and sum reductions are restricted to the
three live fragments. This makes the paired NAX destination bounds-safe without
changing arithmetic. Four experimental-candidate sessions plus one actual-production
session compared BK48 directly with the preceding BK32 route at 100,000 tokens;
all five passed all 102,400,000 compared values and were positive, with a
**1.082579x geometric mean**. The BK32 predecessor had five positive
complete-model sessions against Q-prescaled BQ128 with a **1.072758x** mean.
That BQ128 predecessor's Q-prescale promotion had a
**1.030260x** four-session mean over aligned-Q. The aligned-Q predecessor's four
paired sessions had a **1.022838x** mean over unaligned BQ128. Two fresh direct
comparisons of that predecessor against the pre-NAX
exact-constant champion passed all 102,400,000 values and measured **3.167632x**
and **3.376048x**, a **3.270180x geometric mean**. Three preceding bounds-safe
BK48 exact batch-32 routes exited `0` in **69605.358456 ms**,
**97172.911045 ms**, and **131088.197166 ms**; every run checked all
**3,276,800,000** outputs as finite and used **30.489014 GiB** MPS driver
allocation. The representative three-point median remains **97172.911045 ms**.
One additional same-seed run explicitly using organizer-default matmul
precision `high` completed all outputs finite in **98588.985210 ms**, with the
same output summary and 30.489014 GiB driver allocation. It lies inside the
historical 69.605–131.088-second range and is retained as the preceding
official-default point, not pooled into the differently declared median or
relabelled as an exact observation of the new direct-head route.
At that median the published matrix-work count implies 13.902901 TMAC/s /
27.805801 TFLOP/s. An identical-seed fast/slow pair differed by 1.883306x
despite the same reported output summary, so neither extreme is presented
alone. The historical
pre-repair BK48 source completed in 66.574 seconds, but its declared score-tile
extent was unsafe and that number is retained only as a historical observation.
Four paired safe-versus-prior complete-model sessions were bit-identical and
had a neutral 0.998777x geometric mean; cross-session absolute times are not
attributed to the repair. Earlier BQ64 observations remain 81.266 and 160.192
seconds; their cross-session absolute ordering is not a causal tile comparison.
These rates are not called MFU: the
organizer's final formula/weights and a supported Apple peak-fp32 denominator
are not available. The earlier 416.036-second exact run is a cross-session
historical observation, not the causal promotion comparison.

## Challenge and measurement contract

The task is to accelerate the supplied causal Transformer layer without making
its output unacceptably different. The current PyTorch harness accepts an
element when `abs_error <= 0.002 OR abs_error <= 0.02 * abs(reference)` and
requires the output shape to match. Workshop evidence says entrants optimize
for their own device, float32 is the required baseline/input/output precision,
internal quantization is allowed, compilation and first-run time may be
excluded, and judges will use the submitted device-specific report rather than
rerun every implementation. The exact official per-case weighting, MFU
definition, bandwidth adjustment, padding policy, and aggregation remain
unpublished; local latency is therefore reported as evidence, not converted
into an official score. Source reconciliation is in
[`docs/OFFICIAL_STATEMENT_NOTES.md`](docs/OFFICIAL_STATEMENT_NOTES.md) and
[`docs/SUBMISSION_CONTRACT.md`](docs/SUBMISSION_CONTRACT.md).

Timing used MPS synchronization at each host boundary, warmup before measured
runs, alternating baseline/candidate order, fixed seeds and inputs, and medians.
Authoritative commands now explicitly set float32 matmul precision to the
organizer default `high`; a complete 2.513080x point and an exact 98.589-second
stress-row completion validate that run contract. A three-setting MPS screen
rejected `medium`, while cross-process `highest`/`high` differences were not
treated as causal.
The four static production routes were also revalidated directly against their
current eager controls under `high`: cases 2/3/9/12 measured
1.054495x/1.201002x/1.044573x/1.172183x, with all 28 explicit trials and four
continuity checks passing. Case 9 retains the smallest margin and its disclosed
variance caveat.
Candidates were tested for correctness before timing and were retained only
when same-session measurements repeatedly had a favorable sign. Compilation,
source generation, and the first launch were outside the reported measured
window.

## Result machine

| Item | Recorded value |
|---|---|
| Host | MacBook Pro `Mac17,9` |
| SoC | Apple M5 Pro; 18-core CPU; 20-core integrated GPU |
| Memory | 64 GB unified memory |
| Vendor bandwidth | 307 GB/s (specification, not a local measurement) |
| OS | macOS 26.6.2, Darwin 25.6.0, arm64 |
| Framework | PyTorch 2.8.0 for the retained solution |
| Accelerator | MPS/Metal 4 |
| CUDA/ROCm | Not present; not required for the declared Apple target |

The complete hardware, compiler, runtime and package inventory is in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md).

## Retained implementation

The dispatcher in
[`solution/optimized_transformer.py`](solution/optimized_transformer.py)
combines narrowly gated improvements:

1. classify an all-valid mask once and avoid redundant masking in every layer;
2. pack Q/K/V projections into one linear operation where that route won;
3. use PyTorch scaled-dot-product attention for reference-feasible float32
   shapes, retaining explicit organizer arithmetic for half dtypes where SDPA
   failed the exact predicate;
4. use generated/adapted Metal fused residual-plus-LayerNorm kernels for the
   measured all-valid float32 width-1024 route and exact cases 6, 7, 10 and 13;
5. for the exact published float32 case 8, use cached fp16 weights and fp16
   inputs/results for Q/K/V, attention-output, and both FFN projections while
   retaining fp32 residuals, normalization, GELU, and returned output; and
6. for the exact published float32 case 6, similarly narrow Q/K/V and
   attention-output projection while retaining its FFN and outer arithmetic in
   fp32, and fuse residual addition with the following LayerNorm using the
   measured 128-thread kernel only when every token is valid;
7. apply that narrower half-attention boundary to the exact published float32
   case 13, where its longer sequence makes the conversion-free pairing
   especially valuable; and
8. apply the same narrow boundary to exact float32 cases 1, 4, 5, 7, 10 and
   11 after independent all-valid/padded correctness and repeatability gates;
   and
9. use decomposed exact routes for float32 cases 2, 3, 9 and 12: complete
   half-dense, QKV-only, FFN-only and QKV-only respectively, with every
   half-linear result converted immediately back to fp32; and
10. for exact all-valid float32 cases 2, 3, 9 and 12 under MPS inference,
   lazily trace and freeze a static graph after weights and device are final;
   add inference optimization only for case 3, keep the shared graph slot
   outside module registration and invalidate it on training, weight or
   device/dtype changes; and
11. for long causal head-dimension-64 shapes, run packed Q/K/V, attention-output,
   and FFN projections in fp16 using lazy non-persistent caches, then process
   one batch item at a time so no quadratic score tensor is created. The main
   attention route is Apple's MLX NAX 256x32 register-tiled Metal 4 template
   with sixteen SIMD groups, fp32 MMA accumulation, fp32 online-softmax state,
   query-only padding to enable the aligned-query specialization, and a
   one-time resident-Q prescale that removes the ordinary per-score scaling
   loop.
   Attention-output and FFN linear results return immediately to fp32;
   residuals, LayerNorm, and GELU remain fp32. If Metal 4, the compiler, or the
   custom bridge is unavailable, the code retries the preceding exact-constant
   triton-msl-derived fast-exp shader and ultimately refuses an unsafe
   quadratic fallback.

The bounded float32/float16 attention and fused-normalization sources are
derived from MIT-licensed `triton-msl` commit
`182c1820fd24a836d565e1da842f28414de64084`. The licence is retained in
[`solution/third_party/triton-msl-LICENSE`](solution/third_party/triton-msl-LICENSE),
and [`scripts/verify_solution_provenance.py`](scripts/verify_solution_provenance.py)
independently regenerates the base shaders and verifies the asserted fast-exp
transform. The NAX source is copied from MIT-licensed Apple MLX commit
`3f0bd54ff0c0af5b88530191d5df31010ce54fcd`; all eight headers and the retained
licence are byte-matched by the same verifier.

## Whole 14-row progress scorecard

No organizer score has been issued because nothing has been submitted or
judged, and the organizer has not published the numerical MFU formula or row
weights. The complete current result therefore has two non-interchangeable
measured parts:

| Published rows | Current measurable result | Comparison boundary |
|---|---|---|
| 1–13 | Organizer-default `high` points 2.513080x/2.523037x/2.508106x; 2.514741x arithmetic session mean; all 117 fresh float32 comparisons pass; preceding 117/117 all-dtype matrix and exact cases 2/3/9/12 statically gated | Candidate versus explicit organizer reference; synchronized local timing; earlier process-default points and older champions retained without pooling settings |
| 14 | Current safe BK48 attention plus BM32/BN512 direct-head QKV has a 1.019126x four-session mean over the preceding projection/layout route and all four S100000 outputs are bit-identical; the current route's contention-controlled exact B32 runs are 54.269253125/51.895878706/52.002793707 s (median 52.002793707 s, max/min 1.045733389), with all outputs finite and zero monitored contenders; the preceding route's organizer-default 98.588985210 s point remains separately labelled | No feasible explicit full-shape reference; S8192 explicit reference, scaled explicit checks and full-length continuity; earlier incomplete and rejected promoted-route observations remain disclosed but are not pooled into the final set |

For engineering progress on row 14, the newest direct evidence is the
four-session **1.019126x direct-head-QKV-over-prior** complete-model mean. The
attention route separately retains the five-session **1.082579x BK48-over-BK32**
mean. The BK32
predecessor has the five-session 1.072758x BQ256-over-Q-prescaled-BQ128 mean;
the BQ128 predecessor has a four-session 1.030260x Q-prescale-over-aligned mean;
the earlier chain retains a two-session 3.270180x paired mean against pre-NAX,
a four-session 1.022838x aligned-over-unaligned mean, and the historical
four-session 1.029597x BQ128-over-BQ64 mean. Historical exact observations are
81.266 seconds for BQ64, 115.640 seconds for unaligned BQ128, 146.616 seconds
for aligned-Q, 116.184 seconds for Q-prescaled BQ128,
112.863 seconds for BQ256/BK32, 66.574 seconds for historical pre-repair
BQ256/BK48, and 69.605/97.173/131.088 seconds for the preceding bounds-safe
BQ256/BK48, whose median is 97.173 seconds. Different-session
absolute times are disclosed, not used as causal tile evidence. This is a real
end-to-end development improvement, but it is not an official baseline speedup
and cannot be mixed with rows 1–13 into an official suite score.

## Rows 1–13: synchronized float32 latency

| Case | Baseline median ms | Solution median ms | Speedup |
|---:|---:|---:|---:|
| 1 | 7.687604 | 3.127270 | 2.458247x |
| 2 | 3.180333 | 1.444104 | 2.202288x |
| 3 | 1.639146 | 0.857208 | 1.912192x |
| 4 | 2.798916 | 1.220146 | 2.293920x |
| 5 | 15.412250 | 6.117875 | 2.519216x |
| 6 | 1533.478355 | 648.608416 | 2.364259x |
| 7 | 5.256083 | 1.683646 | 3.121846x |
| 8 | 116.806041 | 38.183896 | 3.059039x |
| 9 | 4.353876 | 2.188250 | 1.989661x |
| 10 | 5.514895 | 2.675271 | 2.061434x |
| 11 | 19.783437 | 5.830937 | 3.392840x |
| 12 | 1.664249 | 0.873146 | 1.906038x |
| 13 | 327.414355 | 71.831583 | 4.558084x |

This table is the earlier seed-8635 **process-default** session whose geometric
mean was **2.517673x**. Each row ran three fresh explicit accuracy trials
followed by two warmups and four alternating five-sample timing rounds. These
synchronized local values are point estimates, not confidence intervals or an
official score. Two matching process-default sessions measured **2.499365x**
and **2.441356x**, making that historical three-session arithmetic mean
**2.486131x**. Those process-default observations are retained as historical
evidence and are not pooled with the later organizer-default `high` result.

The current reportable repeated setting is the organizer-default `high` set:
**2.513080x**, **2.523037x** and **2.508106x**, with arithmetic mean
**2.514741x**, geometric mean **2.514733x**, spread **0.595334%**, and 117/117
fresh float32 comparisons. The preceding champion's older points also remain
disclosed; controlled per-case comparisons, rather than cross-session suite
subtraction, justify promotions. Exact commands,
repeatability and complete score evidence are in
[`docs/SOLUTION_MILESTONE_97_2026-08-30.md`](docs/SOLUTION_MILESTONE_97_2026-08-30.md)
and [`docs/SOLUTION_MILESTONE_100_2026-08-30.md`](docs/SOLUTION_MILESTONE_100_2026-08-30.md).
The current case-9 half-output promotion and score sessions are in
[`docs/SOLUTION_MILESTONE_106_2026-08-30.md`](docs/SOLUTION_MILESTONE_106_2026-08-30.md).

## Stress-row evidence

The retained precision change passed six explicit-reference comparisons at
`B=1,S=8192` (three unpadded and three prefix-padded) and one `B=2,S=8192`
indexing check over 16,777,216 outputs. Four 100,000-token comparisons against
the scaled-reference-validated bounded champion had zero failed elements; the
largest difference was `0.0000286102295`. These full-length comparisons prove
continuity with the validated bounded route, **not equality with an explicit
100,000-token organizer reference**.

The fast-exponential change was retained after three independent paired target
sessions showed speedups of `1.060994x`, `1.010613x`, and `1.012754x` (the last
with balanced first/second ordering), for a session geometric mean of
`1.027860x`. The exact-shape command was:

```bash
PYTHONPATH=. .venv/bin/python experiments/run_case14_solution.py \
  --batch-size 32 --seq-len 100000 --dtype float32 --seed 1743
```

It exited successfully with the exact values summarized above. Absolute times
from different thermal sessions are not attributed to individual changes; only
paired same-session ratios authorize promotion.

The later half-QKV projection passed four target-length paired sessions with
speedups `1.022671x`, `1.066599x`, `1.056514x`, and `1.047601x`. Its scaled
explicit-reference checks covered three unpadded and three prefix-padded seeds,
and a batch-two chunking check passed all 16,777,216 outputs. After local MPS
access was restored, its exact batch-32 command exited `0` in 421.834 seconds,
used 29.979 GiB driver allocation, and reported all outputs finite. This
absolute time is not compared causally with the 362.432-second historical run;
only same-session paired ratios authorize attribution.

The subsequent output/FFN precision candidate first passed an isolated gate,
then a separate production integration. Fresh scaled checks covered three
unpadded and three prefix-padded seeds; batch-two indexing passed all
16,777,216 outputs. Three production target sessions passed all 102,400,000
comparisons and measured `1.028448x`, `1.033322x`, and `1.009851x`, for a
`1.023824x` geometric mean. That milestone's exact batch-32 command exited `0` in
441.937 seconds, used 30.489 GiB driver allocation, and reported every output
finite. Its slower absolute time versus earlier full runs is not used to
override the controlled paired evidence.

The latest change binds the observed dimensions and tensor strides into the
otherwise identical fast-exp Metal source. Fresh unpadded, padded, and
batch-two production gates passed; six full-length sessions were bit-identical
to the preceding champion. Three production A/B sessions measured `1.024330x`,
`1.012527x`, and `1.024917x`, for a `1.020575x` geometric mean. The exact
batch-32 command then exited `0` in 416.036 seconds with all outputs finite and
30.489 GiB driver allocation.

The next structural change used Apple's current MLX NAX attention template
rather than mechanically enlarging the preceding shared-memory tile. A Metal 4
Objective-C++ bridge lets the exact 64x32, four-SIMD-group template operate on
PyTorch MPS buffers without a framework copy. An explicit 8,192-token whole-
model reference check passed all 8,388,608 values. Four independent 100,000-
token continuity seeds each passed all 102,400,000 values. Three production
A/B sessions measured 3.685709x, 3.291094x, 2.883548x, and 2.868742x; the
3.164973x geometric mean authorized promotion. The exact batch-32 command then exited
`0` in 81.266 seconds with every output finite and 30.489 GiB driver allocation.
A separate repeat also exited `0` with all outputs finite but took 160.192
seconds. This large cross-session variation did not change the paired decision
and neither absolute point is selected as universally representative.

The post-promotion profile matched normal forward bit-for-bit and assigned
89.2767% of its intrusive component sum to bounded attention. A follow-up
precomputed causal-loop bound won three isolated target sessions (1.016524x
geometric mean), but lost its first two production-integration sessions at
`0.978048x` and `0.987847x`; it was removed from the retained solution.

A global same-session checkpoint then compared the accumulated current route
with the older fast-exp-only configuration that once had a lower different-
session exact time. The current route measured 12,201.177563 ms versus
13,331.265854 ms, a **1.092621x** speedup; every current sample was faster than
every old sample, and all target outputs passed the organizer predicate. This
supports the sequential promotion path without treating historical absolute
latency as causal evidence.

## Rejected alternatives and lessons

- A fully MLX-native same-input/same-weight implementation repeatedly missed a
  small number of organizer-checked elements, so it was not timed or promoted
  as a valid solution. MLX was useful as a source of Apple-native techniques,
  not as automatic evidence of correctness.
- Apple's native macOS MPSGraph SDPA operator was bridged directly onto
  PyTorch MPS buffers. It passed the explicit reference and improved 1.095521x
  at S512, but reversed to 0.632802x at S2048. The API also has no causal flag
  and requires an explicit broadcastable mask, which contains `L^2` elements.
  It was rejected before S8192 or Transformer integration. A fresh production
  profile matched all 102,400,000 values and kept bounded attention at 89.8836%
  of the intrusive sum, confirming that the rejection does not expose a dense
  component as the next target.
- mlx-mfa's NAX linear kernel showed a promising 1.223567x native MLX result
  after padding its invalid M100000 tail. A direct PyTorch MPS bridge then
  tested all eight upstream tiles on M100000/N1024/K1024. Every one of the
  819,200,000 compared values was bit-identical to the current half linear,
  but all tiles lost at 0.882660x–0.968878x. The favorable native signal did
  not survive the production framework boundary, so no Transformer path was
  changed.
- Generalizing that direct bridge to the actual packed-QKV width
  `M100000/N3072/K1024` passed all 307,200,000 values bit-identically and
  improved the best prior tile to 1.062929x. Packed QKV is only about 3.7% of
  the current profile, so the optimistic ~0.22% whole-row value missed the
  predeclared advancement threshold and was rejected before integration.
- A follow-up included the previously unmeasured layout work: each NAX linear
  SIMD-column group stored directly into contiguous Q/K/V head-major backing
  storage, eliminating the packed token-major output and three materializations.
  After repairing the final partial-row bound, all 307,200,000 exact S100000
  Q/K/V values were bit-identical. The best complete boundary improved
  1.290382x, but at the refreshed 4.0936% component share its optimistic whole-
  row projection is only 1.009298x. It missed both advancement gates and was
  not integrated.
- After larger structural alternatives closed, that direct-head boundary was
  nevertheless tested through the complete production Transformer. S8192
  passed the explicit organizer reference, S32768 and four independent
  S100000 sessions were champion-bit-identical, and the target sessions were
  balanced between champion-first and candidate-first order. Their paired
  geometric means were 1.009923x/1.011522x/1.012118x/1.003220x: a balanced
  `1.009190x`, with 28/36 positive pairs. It remains below the predeclared 1%
  whole-model gate, so the fourth less-favorable session triggered rejection
  and the candidate remains experiment-only.
- Safe, relaxed and fast Metal math modes were then compiled on that identical
  direct-head kernel. All three were bit-identical over 100,663,296 S32768
  values. Relaxed/fast median ratios were `0.993151x`/`0.990938x`, with only
  5/9 and 3/9 positive pairs; positive paired geometric figures were driven by
  safe-mode outliers. Both modes were rejected before target reintegration.
- The five missing direct-head NAX linear tiles were then tested at exact
  S100000; every tile was bit-identical over 307.2 million values. A final
  same-process, position-rotated tournament ranked all eight. Tile
  `BM64/BN128/BK256/WM2/WN4` led the retained experimental tile by only
  `1.012011x` median and `1.007552x` paired (6/8 positive). At the approximately
  4.1% QKV boundary, that adds only about 0.03–0.05% whole-row value, so the
  complete tile family was closed without renewed integration.
- Four legal geometries outside the published eight-tile list then reopened
  only the near-threshold boundary. BM256/BN32 compiled but failed more than
  80 million exact values. BM32/BN512 was bit-identical and improved the
  projection boundary `1.068162x` with 10/10 positive pairs. It passed S8192
  explicit and S32768 continuity gates, then produced four positive balanced
  S100000 whole-model sessions at a `1.019126x` mean. This supersedes the
  earlier direct-head rejection and is retained with a fail-closed prior
  projection fallback.
- Layer-major execution was tested as a full-batch structural alternative to
  the retained one-complete-item-at-a-time stress schedule. It preserved the
  fused residual/norm path, passed the S8192 explicit organizer reference and
  was bit-identical over 134,217,728 B4/S32768 values. All four bounded pairs
  lost, with a `0.993868x` mean, while the schedule required one additional
  full-batch state workspace. It was rejected before S100000/B32 allocation.
- Transposing the safe-BK48 Metal threadgroup grid from `(query block, head)`
  to `(head, query block)` retained identical logical indices and arithmetic.
  It passed S256 explicit correctness and was bit-identical over 33,554,432
  S32768 values, but measured `0.984012x` with 1/8 positive pairs. The exact
  scheduling permutation was rejected before target or whole-model work.
- Fusing tanh GELU into the best NAX linear tile rescued the isolated boundary
  to 1.639707x and passed the S8192 explicit Transformer reference. Its first
  bounded transfer changed from 1.029655x at S8192 to 0.991320x at S32768,
  where all 33,554,432 continuity values still passed. It was stopped before
  target allocation because the end-to-end sign reversed and plausible target
  value was below one percent.
- Plain tracing of the current packed-half case-9 graph changed sign in four
  of five sessions and averaged 0.990331x versus freeze-only production.
  Inference optimization measured 0.628709x. Both passed correctness, but the
  established freeze-only graph remains the measured choice.
- The source-supported fp16-MMA-accumulator mode passed shorter probes but
  failed **12,066,514/102,400,000** outputs at 100,000 tokens; it was rejected
  before target timing.
- Moving the packed QKV cast was arithmetic-identical but changed timing sign
  across two target sessions (`1.001791x`, then `0.991499x`), so it was rejected.
- A Philip-Turner-inspired SIMD-local reduction passed correctness but changed
  timing sign in its third balanced session (`1.006474x`, `1.013785x`, then
  `0.996688x`), so it was rejected.
- Keeping packed Q/K/V as non-contiguous head-layout views was bit-identical to
  the champion, but two target sessions were `0.983998x` and `0.989380x`.
  Repeated strided K/V traversal outweighed the removed one-time layout copies.
- Token-major bounded output with contiguous Q/K/V was also bit-identical, but
  three target sessions changed sign (`0.988345x`, `1.024041x`, `1.001324x`).
  Its sub-percent session average did not justify promotion.
- A direct 64x64 attention tile is impossible under the measured 32,768-byte
  Metal threadgroup-memory limit because it needs 42,752 bytes. A 30,464-byte
  64x32 rearrangement compiled, but failed 2,358,619/8,388,608 scaled outputs
  and was rejected before timing.
- Replacing the repeated causal break with a precomputed loop bound was
  bit-identical and positive in three isolated target sessions, but reversed
  twice after production integration. Source-level simplification did not
  guarantee stable Metal code generation, so it was rejected.
- Compiling the retained NAX kernel with Metal fast math was bit-identical and
  had a 1.082003x geometric mean in three isolated target-attention sessions.
  In three complete-model sessions it changed sign once and fell to a
  1.007096x geometric mean, so Apple's safe default remains retained.
- Retesting fast math after one-time Q prescaling produced 1.019028x at S32768
  but only 1.006022x with overlapping samples at S100000. The changed kernel
  did not rescue the compiler mode, and it was stopped before complete model.
- Cooperatively staging the common BK32 K/V block for all eight SIMD groups in
  8 KiB of threadgroup memory was bit-identical after correcting the probe's
  linear-thread indexing, but measured only 0.730131x at S32768. The two
  cross-group barriers per key block overwhelm the avoided device reloads.
- After BQ256 promotion, BQ320/BQ384/BQ448/BQ512 all compiled and remained
  bit-identical at S32768, but achieved only 0.906141x, 0.849217x, 0.826393x
  and 0.903588x. BQ256 is the measured optimum in this wider-tile family.
- Under BQ256, BK128/BK256 collapsed to 0.357243x/0.142238x. BK64 was valid and
  1.057197x in isolated target attention, but two complete-model sessions
  reversed to 0.975532x and 0.905780x. BK32 was therefore retained at that
  milestone; this did not test the later BK48 gap candidate.
- Closing the remaining even-fragment gap after BK48 promotion, BK96 passed the
  explicit S256 reference and all 33,554,432 S32768 values versus current
  production. It measured only 0.719844x with complete sample separation and
  was rejected before target allocation.
- Filling that gap with BK48 produced five positive 100,000-token complete-
  model comparisons over BK32: `1.057095x`, `1.081094x`, `1.072801x`,
  `1.115832x`, and `1.086934x` (geometric mean `1.082579x`). The fifth used the
  actual production dispatcher. The production S8192 explicit-reference gate
  passed all 8,388,608 values but measured `0.990628x`; BK48 is intentionally a
  stress-length choice, not a universal short-sequence optimum.
- Generalizing the pinned kernel to TQ2/TQ4 at fixed BQ256 reduced the launch
  to 8/4 SIMD groups and remained bit-identical, but measured only 0.981985x
  and 0.229468x. Production retains one query fragment per group.
- Restricting half precision to the QK tensor-op destination while immediately
  converting scores back to fp32 still failed 1/262,144 values at S256 (max
  absolute error 0.00221252441). It was rejected before timing.
- Duplicating the key loop into branch-free off-diagonal and always-masked
  causal-tail sections was bit-identical but only 0.767340x at S32768. The
  compact uniform-branch loop schedules better and remains retained.
- Writing NAX output directly into token-major backing storage preserved layout
  through fp16-to-fp32 conversion and was bit-identical. Its isolated S32768
  attention result was 1.012332x, but complete S100000 sessions were 0.998970x
  and 1.001798x. The apparent merge-heads saving is not a real whole-model win.
- Crossing TQ1/TQ2 with BK32/BK64 showed that the two independent parameters do
  not compensate for each other. All four candidates passed S32768 correctness,
  but TQ2/BK64 fell to 0.389657x; its multiplied per-group fragment state is a
  scheduling/register burden, not useful reuse.
- Removing the no-memory SIMD-group barrier executed before every P@V operation
  preserved all S32768 values but measured only 0.980449x. The marker affects
  compiler/NAX scheduling even without a memory fence and remains retained.
- Narrowing only the normalized probability tile to half before P@V retained
  fp32 softmax/output state and passed all S32768 values, but measured 0.937760x.
  Conversion and the half-input operand path are slower than mixed fp32-P/half-V.
- Retesting view-only packed Q/K/V through the promoted NAX bridge passed the
  S8192 explicit reference but collapsed to 0.395871x. Repeated strided K/V MPP
  loads make the one-time contiguous head-layout copies mandatory on this path.
- Reversing causal query-block grid mapping was bit-identical and improved
  isolated S100000 attention by 1.022960x, but complete target sessions reversed
  to 0.987753x and 0.998877x. Existing grid parallelism already hides the tail.
- Removing BQ256's 96-row Q padding restored MLX's partial-query load/store
  path and was bit-identical at S100000. All seven isolated attention pairs
  favored it, but the gain was only 1.007502x and the complete-model gate
  reversed to 0.991552x. The aligned route remains retained.
- Re-screening case-8 width-1024 fusion directly against production block 512
  rejected blocks 32/64/128/256/1024 at 0.987804x/0.997952x/0.997575x/
  0.994456x/0.999275x. All candidates were correct; the initial apparent
  separate-session differences were thermal drift, not a new optimum.
- Incremental half-FFN screening on exact short cases 1/4/5/7/10/11 rejected
  five rows immediately at 1.006236x or below. Case 1 began at 1.043817x and
  had a 1.020860x four-session mean, but its final session reversed to
  0.990702x. The existing attention-only precision boundary remains retained.
- A clean three-repeat profile assigned 90.2574% to attention. Split-work was
  not built: 6,256 existing threadgroups leave only a 0.637147% optimistic task-
  granularity ceiling, while two-way partial write/read traffic alone has an
  11.017733 ms two-layer bandwidth floor versus an 11.504913 ms ceiling.
- The current BK48 route's fresh three-repeat profile matched normal forward
  over all 102,400,000 values and assigned 89.6899% to attention. Re-screening
  BQ224/BQ240/BQ272/BQ320 under BK48 produced only 0.958420x/0.949920x/
  0.865394x/0.856086x versus BQ256. All were bit-identical and rejected.
- Repaired fast-math comparison on the actual BK48 route was bit-identical at
  S32768 but measured only 1.002708x with overlapping samples. It was stopped
  before target allocation; Apple's safe compiler mode remains retained.
- Halving BK48 query groups through TQ2 reduced repeated K/V loads but doubled
  per-group state. It remained bit-identical over 33,554,432 S32768 values and
  collapsed to 0.377209x. The candidate was rejected before target allocation.
- BK48 source audit found that the paired Q@K loop wrote a fourth score
  fragment while only three were declared. Rounded-and-masked, legal single-
  tail, scratch and discard repairs measured 0.984667x/0.977030x/0.970060x/
  0.977506x. Rounded storage with reductions restricted to three live columns
  measured 1.005325x and was retained. Four S100000 complete-model sessions
  were bit-identical to prior production and had a neutral 0.998777x mean.
- Pairing reduction fragments into legal K32 MPP operations did not improve the
  safe route. Q@K's wider cooperative-input layouts failed the first explicit
  gate and were not timed. The probability×V form was bit-identical over all
  33,554,432 S32768 values but only 0.958437x as fast as production; wider
  operand setup/register cost exceeds the saved K16 operation.
- PyTorch `2.15.0.dev20260829` was tested in an isolated official-nightly
  environment rather than assumed to be faster. It passed all 39/39 ordinary
  accuracy trials and the safe-BK48 S8192 boundaries, but its 13-row geometric
  mean was 2.142523x versus the retained PyTorch 2.8 champion's 2.427904x;
  case 3 regressed to 0.935455x. Its small causal SDPA still profiled through
  `aten::_scaled_dot_product_attention_math_for_mps`. The nightly and its
  temporary C++20 bridge compatibility edit were rejected.
- Splitting causal queries between BK32 for an early prefix and BK48 for the
  tail avoided all Q copies and wrote two disjoint windows of one output. S512
  explicit and all S32768/S100000 continuity gates passed. Broad-screen target
  medians initially suggested up to 1.028442x, but corrected alternating-order
  sessions measured only 0.997634x and 1.001641x paired geometric means. The
  thermal/order artifact was rejected before whole-model integration.
- Retesting Q@K and P@V loop interchanges under the exact current BQ256/BK48
  state produced bit-identical S32768 output but no win: 0.996563x for Q@K,
  0.997218x for P@V, and 0.971830x combined. The retained MLX operation order
  remains faster despite equivalent arithmetic.
- `torch.compile` was tested after excluding compilation/first calls and
  removing the all-valid mask graph break. It remained correctness-valid but
  default mode measured only 0.778558x/0.869990x/0.218243x paired on cases
  3/13/6. Case-3 reduce-overhead/max-autotune were 0.635577x/0.651111x. Eager
  MPS execution is retained.
- Tanh-approximate GELU passed 32/32 case-3 all-valid/padded trials and began
  with five positive sessions (1.014117x mean), but a fresh production-aware
  session reversed to 0.989695x. The proposed exact-case edit was fully reverted
  and exact GELU remains retained.
- PyTorch's documented MPS fast-math/prefer-Metal process switches produced no
  material case-6 candidate change (470.493/470.288/470.487 ms) and default
  also had the best case-3 candidate median. Neither switch enters the run
  contract.
- Guarding the online-rescaling exponential when the running maximum stayed
  unchanged was bit-identical but only 1.000404x at S32768 with overlapping
  samples. Unconditional `fast::exp2` remains simpler and equally fast.
- Also skipping factor-1 output-fragment multiplies remained bit-identical but
  reached only 1.002366x with overlap. Conditional rescale shortcuts do not have
  a target-worthy margin on this compiled kernel.
- NAX BK16 variants at BQ32, BQ64, and BQ128 compiled but each failed
  128,310/131,072 values at the first explicit gate. MLX uses BK16 in a
  different wide-head split path; no invalid head-64 variant was target-timed.
- Binding scalar NAX shape constants was bit-identical but slower (0.970683x
  median). Binding 64-bit strides, alone or with scalars, reproducibly crashed
  Metal pipeline compilation via `XPC_ERROR_CONNECTION_INTERRUPTED`; those
  variants never launched, and a fresh production build remained healthy.
- MLX's distinct paired-SIMD head-dimension split kernel compiled and passed all
  target values at BD64, but its exchange/redundant-softmax overhead measured
  only 0.812349x versus aligned BQ128. It is retained only as research evidence.
- NAX fp16 accumulation/online state compiled but failed 1/131,072 values at
  the first S128 explicit gate (max abs 0.0029296875), so the invalid candidate
  was never allocated or timed at target length.
- A narrower half P@V output-fragment candidate retained fp32 scores and
  online-softmax state, but Metal 4 rejected its mixed float-probability/half-
  destination tensor operation as an unsupported type before launch.
- Removing the ordinary NAX kernel's final no-memory threadgroup barrier was
  bit-identical at S100000 but measured only 0.937002x. The apparently redundant
  barrier remains because Metal scheduling/code generation favored it.
- Hoisting the four BD64 query-fragment loads out of the key/value loop was
  bit-identical over all 102,400,000 target values, but measured only 0.982704x.
  Query loads were not a useful isolated bottleneck, and the extra residency
  likely increased register pressure or worsened scheduling.
- Replacing Apple's four-way/full BD64 head-loop unroll with two-way or disabled
  unrolling preserved all 33,554,432 bounded values but fell to 0.822039x and
  0.793031x. Both candidates were stopped before target allocation.
- Processing two or four published stress-row batch items together was
  bit-identical over 134,217,728 bounded outputs, but measured 0.998413x and
  0.977581x versus one-item chunking. The larger live state provided no useful
  extra attention throughput, so the exact-B32 memory-safe synchronization
  boundary remains retained.
- After Q prescaling changed the BQ256 interaction, BQ160/BQ192/BQ224 found
  only 0.707511x/0.764056x/0.878019x versus BQ256. The actual immediate
  BQ240/BQ272 neighbors were later measured at 0.938162x/0.887513x; all passed
  correctness and were rejected before target allocation. Together with
  BQ320–BQ512, the tested family and both immediate neighbors favor BQ256.
- Narrowing only the high-frequency `exp2(score-row_max)` evaluation through
  half precision passed ten explicit seeds and two bounded continuity gates,
  but separate sessions reversed from 0.984851x to 1.016999x. It was rejected
  without a target-length run.
- Interchanging the fully unrolled P@V loops preserved all 33,554,432 bounded
  values but measured 0.962052x. The retained output-dimension-major instruction
  order is faster than consuming both dimension pairs per key fragment.
- An early 16-query-row tile adaptation compiled but failed nearly all outputs;
  it never entered the solution.
- Replacing SDPA inside the four retained exact static graphs with traced
  explicit QK/causal-mask/softmax/PV algebra passed 20/20 organizer comparisons,
  but cases 2/3/9/12 measured only 0.781941x/0.783006x/0.881829x/0.789353x.
  Static mask folding does not overcome explicit attention cost on MPS; the
  production graphs retain SDPA.
- Folding `scale*M_LOG2E` into the packed-half Q weight and bias was closed
  analytically. The retained NAX path already scales Q only once outside the
  key loop; the proposed change removes at most 6,553,600,000 scalar multiplies
  across row 14, just 0.000485% of its counted matrix MACs, while changing the
  half-rounding order. Its deliberately optimistic ideal bound is only
  1.000004851x, so no candidate or target allocation was justified.
- Sage-style block-int8 K was numerically viable but performance-invalid.
  Five S8192 whole-model probes passed the organizer reference. An experiment-
  only current-NAX Metal adaptation reconstructed K bit-identically at S32768,
  yet measured 0.778038x before quantization and 0.681579x including it. The
  pinned upstream cooperative Sage kernel measured only 0.19x/0.23x versus its
  fp16 attention at causal D64 S8192/S32768. Conversion, multiplication and
  threadgroup synchronization outweigh halved K storage on this Apple M5 Pro.
- A source-distinct TQ2 experiment moved K/V fragment loads outside the two-
  query loop, achieving actual register reuse without shared memory or barriers.
  It remained bit-identical but measured 0.961426x versus BK32 TQ1 and
  0.937859x versus production BK48. The doubled score/output accumulator state
  still dominates; TQ1 remains the measured mapping.
- Fused residual normalization was tested across all four retained static rows,
  not assumed complementary. Cases 2/3/9/12 passed 60 organizer comparisons,
  but block-64 and block-128 eager-fused routes lost on every row; applying all
  four would multiply the cases-1–13 geometric score by only
  0.959827x/0.960079x. The frozen graphs retain broader value than the isolated
  shader fusion on these small shapes.
- Apple's float32/TF32-capable NAX boundary was instantiated with the current
  safe BQ256/BK48 structure. Five S256 explicit-reference seeds and all
  33,554,432 S32768 continuity values passed, but TF32 measured 168.143500 ms
  versus 96.612875 ms for retained half-input/fp32-accumulate NAX, or
  0.574586x. Capability did not imply a performance win, so no target-length
  or whole-model allocation followed.
- The new MIT `mlx-train-perf` attention source at `cdfce970` was acquired and
  read at its exact revision. It targets training-memory reduction, reports a
  5.3–5.9% training-throughput cost in its measured comparisons, and has only
  one saturated forward tile result at B1/Hq32/Hkv8/N8192/D128/bf16. Its D64
  classic-SIMDgroup route is explicitly provisional. Row 14 is B32/H16/N100000/
  D64 inference and already uses the M5 NAX family, so no build, model
  allocation or unsupported performance claim was made.
- A correctness-only sliding-window ladder tested the last large algorithmic
  work-reduction hypothesis. Full-window chunked controls passed, but every
  shorter window failed. A 75%-length window retains 93.75% of causal pairs;
  it failed 102,874/8,388,608 final values against the explicit S8192 reference
  and 24,399/102,400,000 values against the exact-length champion. Even windows
  that removed only about 0.39% of pairs failed through S32768. No approximate
  Metal kernel was written or timed; dense causal attention remains required.
- An optimistic content-sparsity oracle then computed all causal scores and
  retained the exact largest K per query. Full-K controls passed, but at S8192
  exact K=6144 failed 132,908/8,388,608 explicit-reference values and K=7680
  still failed 2,610. Because exact selection is already invalid and its full-
  QK arithmetic ceiling is only 1.0323x at K=0.75N before selection overhead,
  no approximate block selector or sparse Metal kernel was pursued.
- Exact B32/S100000 per-item instrumentation then separated first-use work from
  sustained execution. A no-warm diagnostic completed in 82.973 seconds with
  a 2.412-second item median; a successive one-item-warm diagnostic completed
  in 112.283 seconds with a 3.416-second item median. Both exited zero with all
  outputs finite. The warm-up removed the first-use penalty, but the dominant
  drift remained between processes. Each run also had one isolated stall after
  roughly one minute, coincident with MPS driver allocation falling from
  30.489 to 30.104 GiB and returning. The cause is not inferred without
  counters; stalls remain part of end-to-end time and neither diagnostic is
  pooled with the declared organizer-default point.
- An experiment-only public Metal command-buffer probe then attached directly
  at each native NAX dispatch. On one warmed exact B32/S100000 diagnostic, two
  unique layer buffers per item accounted for 57.203 of 66.570 measured
  seconds (85.93%). The NAX first/last-quarter medians were approximately
  1.560/1.903 seconds, a 1.2197x increase, proving material GPU execution drift
  even without a recorded warning. Completion handlers can perturb scheduling,
  so the 66.570-second value is diagnostic only and does not replace the
  organizer-default point.
- The BK48/BK32 choice was then retested under sustained shared heat. A B8
  target-length alternating gate passed all 819.2 million values and favored
  BK48 on 8/8 items at a 1.100214x paired mean. A full B32 target-length A/B
  passed all 3.2768 billion values and produced a 1.137625x balanced paired
  mean, but route-first strata were 1.307958x and 0.989475x and both routes
  suffered multi-second stalls. The balanced result strengthens retained BK48;
  the two-model thermal/memory process is not substituted for one-model
  scorecard latency.
- A final public-source refresh acquired the Apache-2.0
  `manishklach/mlx-metal-kernels` checkout at `9fc1d38a`. Its default attention
  is explicitly one thread per row; experimental routes are scalar
  threadgroup/SIMD reductions without `simdgroup_matrix` or Metal-4 NAX, the
  checkout publishes no Apple timings, and its fp16 oracle uses a much looser
  absolute tolerance. It was closed by source audit without build or tensor
  allocation and contributed no production code.
- Because exact latency drift is now measured rather than hypothetical, the
  final result protocol is automated and fail-closed. It runs three separate
  identical-seed B32/S100000 processes, requires 60 consecutive seconds
  without an observed external Python, Node or Codex process at or above 50%
  CPU before every
  run, polls once per second throughout each measured run, excludes exactly one compile-warm
  item, records all 32 item timings, waits five minutes between processes and
  trims no sample. Memory, swap and process state are recorded; power state is
  neither inspected nor controlled. A claim additionally requires every output
  finite, zero monitored runtime contenders and
  `max_elapsed/min_elapsed <= 1.15`. The pre-run window is an initial-idle
  check, not a prediction that no later job will start: any process that appears
  during measurement invalidates that run. In execution mode, contention in
  the first snapshot enters the same quiet-window monitor instead of creating
  a launch-time race; missing or failed observability queries still block
  immediately. An irrecoverable partial max/min spread above 1.15 now stops the
  set before another allocation. The 98.589-second preceding-route
  organizer-default point remains unchanged until this set completes.

The source-by-source accounting is in
[`docs/SOURCE_TO_EXPERIMENT_LEDGER_2026-08-29.md`](docs/SOURCE_TO_EXPERIMENT_LEDGER_2026-08-29.md).

## Quantified convergence gate

The exact-row profile assigns **89.8836%** of measured time to attention and
**4.0936%** to QKV. Applying Amdahl's law to a minimum **3% whole-row** gain
gives explicit reopening thresholds:

- an attention-only change must accelerate attention by at least
  **1.033489578x**;
- a QKV-only change must accelerate QKV by at least **3.466277762x**; and
- a change to only one of the 13 equally aggregated ordinary rows would need
  **1.468533713x** on that row to move their geometric mean by 3%.

These are screening thresholds, not measured wins. They prevent a locally
interesting microbenchmark from consuming final-measurement time when its
best plausible whole-result effect is immaterial. The zero-cost theoretical
ceilings are 9.884939306x for attention and only 1.042683283x for QKV, before
any interaction or overhead.

The retained experiment ledger closes the presently available structural
attention families: complete query/key tile and task-shape sweeps,
compiler/math modes, barrier and loop order, K/V staging and reuse, reduced
probability precision, int8 K/Sage-style routes, MPSGraph/native routes and
the latest relevant public Apple-attention sources. Each alternative either
failed the organizer-reference correctness gate, reversed in the whole model,
or did not clear a credible 3% whole-row projection. The selected direct-head
QKV route's four balanced target-length sessions produced a 1.019126x
whole-model gain; there is no measured path from that evidence to the
3.466278x QKV-component threshold. Rows 1–13 are already faster in every one
of three complete sessions, and the weaker rows' graph, half-precision and
fusion alternatives have also been correctness and whole-model gated.

Production is therefore convergence-frozen with the contention-controlled
final protocol complete. Technical exploration reopens only for new,
exact-causal D64 Apple evidence that passes the explicit reference and has a
credible path beyond the applicable threshold. The frozen decision does not
claim optimality, an official MFU or an official combined score.

## Verification and reproduction

```bash
.venv/bin/python scripts/verify_release_manifest.py
./scripts/acquire_solution_upstreams.sh
.venv/bin/python scripts/verify_solution_provenance.py
PYTHONPATH=. .venv/bin/python -m pytest -q

# Accuracy-only example for one published reference-feasible case:
PYTHONPATH=. .venv/bin/python experiments/benchmark_sdpa_candidate.py \
  --candidate solution --case 1 --device mps --dtype float32 \
  --accuracy-trials 5 --skip-timing

# Safe final stress-shape preflight; this does not execute the benchmark:
.venv/bin/python scripts/run_case14_final_protocol.py
```

Create the environment from `requirements-lock.txt` for the exact tested
18-package resolution. The public setup uses `--require-hashes` and
`--only-binary=:all:` so each target wheel must match its recorded SHA-256.
`requirements-solution.txt` retains only the four direct declarations.
Dependency, wheel and installed-wheel licence metadata are recorded in
[`docs/DEPENDENCY_LICENSES.md`](docs/DEPENDENCY_LICENSES.md).

The earlier fresh-environment audit passed **36 tests** with MPS visible. The
promoted-champion clean export passed **72 tests** at that checkpoint. The
current authoritative suite passes **111 tests**; the final deterministic
sanitized-package checkpoint passes **33 tests with one explicit
missing-organizer-input skip** in public-safe mode and **101/101** after a
checksum-matched authorized attachment is supplied. The final package was also
rehearsed from a wholly new uncached Python environment: all four exact pins
installed, `pip check` passed, MPS was built and available, the packaged
verifier passed, and the full 101-test authorized-input suite rebuilt the
retained native path from a previously nonexistent extension cache. The
provenance verifier regenerated all retained sources,
including the fused-128, fused-64 and fused-32 kernels, exactly, and the official
artifact checksum verifier passed. Large runtimes,
upstream clones, caches and outputs are ignored and are not needed to execute
the committed retained solution. Regenerating and byte-checking provenance
does require the exact ignored checkouts, now reproducibly acquired by the
committed script. The fail-fast promoted direct-head QKV champion source is
commit `622c47c`. The enclosing sanitized release's
`RELEASE_METADATA.json` records its exact content commit, so this report does
not hard-code a packaging commit that would become stale on the next
documentation-only rebuild. The signed `docs/FINAL_ACTION_CHECKLIST.md`
defines the current clean-history and fresh-environment handoff. The original
public-safe trees remained free of organizer Python throughout every
authorized disposable-copy rehearsal.

The final local result identity is also machine-readable in
`docs/CHAMPION_MANIFEST.json`. Its verifier SHA-256 checks six production
inputs, preserves the exact selected tiles and fails if the validated current
route measurement, preceding-route separation or no-official-score boundary
drifts.

## Limitations and publication gates

- No official MFU or final score is claimed; the scoring formula is incomplete.
- Full explicit-reference equality at 100,000 tokens is unavailable because
  the organizer reference is physically infeasible on this machine.
- Long-sequence mask support is deliberately limited to the all-valid or
  organizer-style prefix form; other masks fail closed rather than allocating
  a quadratic fallback.
- The result is device-specific to this Apple M5 Pro and has not been certified
  on another Mac or on CUDA/ROCm hardware.
- The preceding bounds-safe Q-prescaled aligned BQ256/BK48 projection route's three exact
  batch-32 runs completed in 69.605/97.173/131.088 seconds; 97.173 seconds is
  the median, and the 66.574-second pre-repair run remains a historical fastest
  observation. Its promotion evidence is the 1.082579x
  five-session complete-model mean over BK32, while four paired safe/prior
  sessions were bit-identical with a neutral 0.998777x mean. BK32 retains its separate
  1.072758x mean over Q-prescaled BQ128; earlier paired evidence remains
  disclosed rather than multiplied into an invented official score.
- The promoted BM32/BN512/BK256 direct-head QKV route's claimable evidence
  includes four order-balanced S100000 B1 sessions, exact continuity and the
  final contention-controlled B32 set at
  54.269253125/51.895878706/52.002793707 seconds. Earlier individually clean
  incomplete-set runs and contention-rejected observations remain disclosed but
  are not pooled. The preceding 98.588985210-second organizer-default point is
  not relabelled as the promoted route.
- Historical BQ64 exact runs passed in 81.266 and 160.192 seconds. All
  cross-session observations are retained rather than selecting one as
  universally representative.
- The NAX route requires macOS 26, Metal language 4.0, Command Line Tools,
  `ninja`, and an NAX-capable Apple GPU. It compiles during warm-up and falls
  back to the preceding champion if any requirement is unavailable.
- The two organizer attachments have no visible redistribution licence. They
  must remain outside any public release unless an authoritative permission or
  public source resolves that issue.
- Entrant/team attribution, final AI-token totals, public-repository contents,
  the demonstration video and the Devpost text are action-time submission
  items. They are not inferred or completed here.
- No submission, public visibility change, organizer contact, registration
  change, remote push or credential action was performed by this report.

AI assistance, human steering, source use and the current accounting boundary
are disclosed in
[`docs/AI_TOOL_DISCLOSURE_2026-08-29.md`](docs/AI_TOOL_DISCLOSURE_2026-08-29.md).
