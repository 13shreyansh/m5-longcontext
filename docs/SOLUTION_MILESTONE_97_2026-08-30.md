# Solution milestone 97 — exact cases 9/12 frozen graphs promoted

Date: 2026-08-30 SGT

## Outcome

Exact all-valid float32 cases 9 and 12 now use the shared freeze-only static
graph lifecycle. Case 11 was integrated and measured but removed: its five
production sessions averaged only 1.004957x, below a defensible margin.

Case 9 measured `1.011676x`, `1.053686x`, `0.990570x`, `1.040440x`,
`1.042818x`, `1.019250x` and `1.042571x`: six positive sessions, one slight
reversal and a **1.028507x** geometric mean. Case 12 measured `1.060069x`,
`1.215353x`, `1.215610x`, `1.105561x`, `1.184154x` and `1.138621x`: all six
positive, with a **1.151773x** mean. Every frozen output was bit-identical to
its eager control. Case 9's reversal is disclosed; its seven-session aggregate
and complete sample set justify retaining the modest exact-shape route.

The retained exact gates are MPS float32, evaluation/no-gradient, all-valid
mask, `D128/F128/L4`, causal, plus `B64/S128/H1` for case 9 and `B64/S32/H4`
for case 12. Near shapes, padding, other dtypes/devices, training and failures
retain eager production. Training, weight load and device/dtype moves
invalidate the unregistered graph.

## Fresh complete score

Each row ran three accuracy trials, two warmups and four alternating
five-sample timing rounds at seed 8360:

| Case | Baseline median ms | Solution median ms | Speedup |
|---:|---:|---:|---:|
| 1 | 7.768730 | 3.121834 | 2.488515x |
| 2 | 3.030000 | 1.397750 | 2.167769x |
| 3 | 1.503917 | 0.858354 | 1.752094x |
| 4 | 2.847125 | 1.253375 | 2.271567x |
| 5 | 15.471438 | 6.341292 | 2.439793x |
| 6 | 1221.166645 | 519.294125 | 2.351590x |
| 7 | 5.001980 | 1.603020 | 3.120347x |
| 8 | 80.937312 | 29.152375 | 2.776354x |
| 9 | 4.134437 | 2.463646 | 1.678179x |
| 10 | 5.499500 | 2.692584 | 2.042462x |
| 11 | 19.846001 | 5.851959 | 3.391343x |
| 12 | 1.664021 | 0.871730 | 1.908873x |
| 13 | 280.431312 | 51.210645 | 5.476035x |

The geometric mean is **2.473404x**, the highest measured complete point so
far. AC power was at 100%, and macOS reported no recorded thermal/performance
warning. Prior 2.454234x, 2.388913x and 2.427904x points remain disclosed; the
same-session A/B results are the causal promotion evidence.

A fresh matrix passes **117/117 trials** across cases 1–13 and all three dtypes,
largest maximum absolute error `0.0016160011`. The repository passes **64/64
tests**. Safe-BK48 case-14 S8192 passes all 8,388,608 values against BK32
continuity and the explicit reference, with maximum errors `0.000938177109` and
`0.000878304243`. Organizer hashes and both upstream licence/provenance chains
pass.

| Published rows | Verified current evidence | Boundary |
|---|---|---|
| 1–13 | Fresh 2.473404x synchronized point; 117/117 matrix; exact cases 2/3/9/12 static routes independently gated | Point estimate, not official score or confidence interval |
| 14 | Safe BK48; current S8192 continuity and explicit reference pass; exact B32 remains 97.173 s | Full explicit target reference remains physically infeasible |

Official organizer score and MFU remain **unknown**. No submission, push,
publication, organizer contact or visibility action occurred.
