# Milestone 148 — BM32/BN512 direct-head QKV promoted

> **Historical checkpoint — superseded for current status.** This file
> preserves the evidence and decisions available at its timestamp. Its
> pending/provisional, battery/power and then-current test-count wording is not
> the submission status. Use the current `README.md`, `TECHNICAL_REPORT.md` and
> `docs/CHAMPION_MANIFEST.json` for current claims.

Timestamp: **2026-08-30 14:46 SGT**

## Why this family reopened

The complete eight-tile upstream sweep in milestone 147 closed every tile
published by Apple MLX, but the rejected direct-head whole-model route was only
`1.009190x`, immediately below its predeclared `1.01x` gate. Four legal NAX
geometries outside that published list could still change projection
parallelism materially: two BN512 forms and two BM256 forms. The advancement
gate was set before target timing: an alternative needed at least `1.03x` at
the complete projection-plus-layout boundary before renewed whole-model work.

## Exact projection screen

The committed harness compiles all candidates, compares Q, K and V separately
against MPS half `F.linear` plus the current head materialization, rotates order
across ten rounds, and excludes any failing candidate from timing.

```bash
/usr/bin/time -l env \
  TORCH_EXTENSIONS_DIR="$PWD/artifacts/nax-qkv-head-layout-edge-extensions" \
  PYTHONPATH=. .venv/bin/python \
  experiments/benchmark_nax_qkv_head_layout_edge_tiles.py \
  --tokens 100000 --seed 9230 --warmup 2 --repeats 10
```

The final command exited zero. Every valid tile was bit-identical over
`307,200,000` values. BM256/BN32 compiled but failed between `80,736,344` and
`81,749,580` values across repeated exact gates, with maximum absolute error as
large as `264.625`; it was never timed as a valid candidate.

| Tile | Exact | Median ratio vs control | Paired mean | Positive |
|---|---:|---:|---:|---:|
| control BM64/BN256/BK256/WM2/WN4 | pass | `1.000000x` | — | — |
| BM32/BN512/BK256/WM1/WN8 | pass | `1.059113x` | `1.068162x` | 10/10 |
| BM64/BN512/BK256/WM2/WN8 | pass | `0.949539x` | `0.967524x` | 2/10 |
| BM256/BN32/BK256/WM8/WN1 | **fail** | not timed | not timed | — |
| BM256/BN64/BK256/WM8/WN2 | pass | `0.957326x` | `0.968417x` | 2/10 |

The successful screen ran at 52–51% battery while discharging. That state is
not used for absolute latency, but the same-process position-rotated ratio was
large and unanimous enough to clear the predeclared projection gate. The
command used 278,953,984 bytes maximum RSS, 7,262,356,416 bytes peak footprint,
6.508621 GiB MPS driver allocation, zero swaps and 5.70 seconds real time.

## Correctness ladder

Before target timing, the selected BM32/BN512 tile ran through the complete
Transformer at an explicit-reference shape and a larger continuity shape:

```bash
/usr/bin/time -l env \
  TORCH_EXTENSIONS_DIR="$PWD/artifacts/nax-qkv-head-layout-bn512-candidate" \
  PYTHONPATH=. .venv/bin/python \
  experiments/benchmark_case14_nax_qkv_head_layout.py \
  --seq-len 8192 --seed 9240 --skip-timing

/usr/bin/time -l env \
  TORCH_EXTENSIONS_DIR="$PWD/artifacts/nax-qkv-head-layout-bn512-candidate" \
  PYTHONPATH=. .venv/bin/python \
  experiments/benchmark_case14_nax_qkv_head_layout.py \
  --seq-len 32768 --seed 9241 --skip-explicit-reference --skip-timing
```

S8192 passed the organizer reference with `0/8,388,608` failures and was
champion-bit-identical. S32768 was bit-identical over all `33,554,432` values.
Both commands exited zero. Their real times were 22.00 and 11.69 seconds;
maximum RSS was 978,468,864 and 978,092,032 bytes; peak footprint was
10,361,062,504 and 4,107,830,376 bytes; both reported zero swaps.

## Four balanced target sessions

Four independent fixed-seed processes alternated champion/candidate ordering;
seeds 9242/9244 began champion-first and 9243/9245 began candidate-first.

```bash
env TORCH_EXTENSIONS_DIR="$PWD/artifacts/nax-qkv-head-layout-bn512-candidate" \
  PYTHONPATH=. .venv/bin/python \
  experiments/benchmark_case14_nax_qkv_head_layout.py \
  --seq-len 100000 --seed SEED --warmup 1 --repeats 9 \
  --skip-explicit-reference [--candidate-first]
```

| Seed | First route | Paired mean | Median ratio | Positive | Exact continuity |
|---:|---|---:|---:|---:|---:|
| 9242 | champion | `1.011633x` | `1.017851x` | 7/9 | 0/102,400,000 failed |
| 9243 | candidate | `1.021096x` | `1.020042x` | 9/9 | 0/102,400,000 failed |
| 9244 | champion | `1.021071x` | `1.014127x` | 8/9 | 0/102,400,000 failed |
| 9245 | candidate | `1.022742x` | `1.027731x` | 9/9 | 0/102,400,000 failed |

The balanced geometric mean is **`1.019126x`**, the geometric mean of session
median ratios is `1.019926x`, and 33/36 individual pairs were positive. Every
candidate output was bit-identical to the preceding champion. Commands exited
zero in 60.85/52.25/53.30/53.17 seconds real, used at most 978,403,328 bytes
maximum RSS and 10,191,389,848 bytes peak footprint, and reported zero swaps.
Battery state declined from 50% to 44% while discharging; only paired ratios,
not absolute latency, are used for promotion.

## Production promotion and clean integration

The selected generator and Objective-C++ bridge moved under `solution/`.
Production uses it only inside the already bounded, packed-half row-14 route.
Unsupported inputs, compilation failure or launch failure return to the prior
MPS half-linear plus materialization path. The source stays derived from the
pinned Apple MLX MIT headers and adds no new dependency or licence.

Fresh-cache S8192 passed the organizer reference with `0/8,388,608` failures;
fresh-cache S100000 passed `0/102,400,000` champion-continuity failures. The
provenance verifier now asserts BM32/BN512/BK256/WM1/WN8 and the direct
head-major store. Official hashes passed, `git diff --check` passed, and the
expanded suite reported **72 passed**, including fail-closed launch coverage.

The unmodified production runner then completed B1/S100000 under explicit
organizer-default `high` precision:

```bash
/usr/bin/time -l env \
  TORCH_EXTENSIONS_DIR="$PWD/artifacts/nax-qkv-head-layout-bn512-production" \
  PYTHONPATH=. .venv/bin/python experiments/run_case14_solution.py \
  --batch-size 1 --seq-len 100000 --dtype float32 --seed 9252 \
  --matmul-precision high
```

It exited zero with every `102,400,000`-value output finite in 3036.870000 ms,
reported 6.502686 GiB driver allocation, 341,950,464 bytes maximum RSS,
7,321,371,608 bytes peak footprint and zero swaps. The battery was discharging
and had been 29% immediately before the paired run, so this proves dispatcher
completion and output health, not an authoritative absolute latency.

## Whole-suite scorecard

| Published rows | Current evidence | Change |
|---|---|---|
| 1–13 | organizer-default `high` points `2.513080x`/`2.523037x`/`2.508106x`; mean `2.514741x`; 117/117 fresh float32 trials plus preceding all-dtype matrix | unchanged |
| 14 | safe BK48 attention plus promoted BM32/BN512 direct-head QKV; preceding organizer-default exact point `98.588985` seconds remains the declared absolute observation; every prior exact output finite | QKV boundary promoted at `1.019126x` balanced whole-model mean; a new stable-power exact B32 point remains blocked by battery state |

No official MFU or combined organizer score is inferred. The prior exact
absolute point is not relabelled as the new route; the final fail-closed
three-process measurement must succeed under its declared power conditions.
