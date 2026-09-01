# Solution milestone 100 — frozen case-9 packed-half QKV promotion

Date: 2026-08-30 SGT

## Outcome

Exact all-valid float32 case 9 now packs Q/K/V into one internally-fp16 linear
before its frozen graph. The result returns to float32 before the unchanged
output projection; the already-retained half FFN remains separate. Native
float16/bfloat16, padding, near shapes, other devices, training and failures
retain their prior paths.

This revisits an interaction rather than overturning the old evidence. Packed
fp32 QKV inside the frozen graph was bit-identical but measured only
**0.882486x** versus the current separate-QKV frozen route. Packed-half QKV,
however, passed 19 pre-integration explicit trials and five all-positive
sessions at `1.113244x`, `1.184204x`, `1.111454x`, `1.106226x` and
`1.103723x`, a **1.123369x** geometric mean.

Five production-aware sessions then measured `1.091852x`, `1.114902x`,
`1.128260x`, `1.080055x` and `1.114265x`, also all positive, with a
**1.105729x** mean. Their 19 explicit trials passed; the largest recorded
production explicit maximum absolute error was `0.00151133537` and the largest
candidate/current continuity maximum was `0.00125849247`.

## Repaired integration failure

The first broad matrix was **not** a pass. Float32 rows completed, but case-9
float16 and bfloat16 subprocesses aborted in Apple MPS with:

```text
Destination NDArray and Accumulator NDArray cannot have different datatype in MPSNDArrayMatrixMultiplication
```

The exact-config flag was contributing outside its intended float32 route, and
the new return-to-float cast was unconditional on input dtype. Both effects are
now gated by the existing float32 `use_sdpa` predicate. The two failed routes
then passed independently, followed by a fresh fail-fast **117/117** matrix
across all 39 case/dtype combinations at seed 8480. Its largest maximum
absolute error was `0.0016233623` on float32 case 8; native float16/bfloat16
remained bit-identical to the organizer reference in this matrix.

The exact padded case-9 production regression also passes. Case-14 safe-BK48
S8192 passes all 8,388,608 values against BK32 continuity and the explicit
reference, with maxima `0.000955283642` and `0.000882714987`. Organizer hashes,
both provenance/licence chains and all **64/64 tests** pass.

## Current complete score

Three current-champion sessions used three accuracy trials per row, two warmups
and four alternating five-sample timing rounds:

| Case | Seed 8460 | Seed 8500 | Seed 8510 |
|---:|---:|---:|---:|
| 1 | 2.406924x | 2.489013x | 2.461246x |
| 2 | 2.158673x | 2.343615x | 2.200047x |
| 3 | 1.774162x | 1.895626x | 2.096665x |
| 4 | 2.065837x | 1.995930x | 2.301047x |
| 5 | 2.522914x | 2.438472x | 2.467896x |
| 6 | 2.459639x | 2.353244x | 2.359312x |
| 7 | 3.228220x | 3.227274x | 3.076537x |
| 8 | 2.604480x | 3.143704x | 3.014005x |
| 9 | 1.922162x | 1.840356x | 1.878273x |
| 10 | 2.087044x | 2.085919x | 2.047777x |
| 11 | 3.362956x | 3.355157x | 3.321056x |
| 12 | 1.790164x | 1.859329x | 1.975825x |
| 13 | 5.470050x | 5.011718x | 4.718354x |
| **Geometric mean** | **2.475722x** | **2.506610x** | **2.523131x** |

The current three-session aggregate is **2.501744x**. Every row in every
session passed correctness and remained faster than its same-session explicit
baseline. The preceding champion's comparable three-session aggregate was
2.481772x; direct case-9 comparisons, not the cross-session difference, remain
the causal promotion evidence.

These are local synchronized results, not an official score or MFU. No
submission, push, publication, organizer contact or repository-visibility
action occurred.
