# Track 3 public-release change audit

Official update timestamp: **2026-08-27 18:25 SGT**

Audited: **2026-08-28 15:06 SGT**

This audit compares the public Lark statement and attachments against the
byte-preserved early-bird snapshot acquired on 2026-08-26. It is
preparation-only: no participant implementation, optimization, submission,
organizer contact, or large benchmark run was performed.

## Official update notice

The same official Lark document now displays `Last updated: Aug 27`. At the
start of Track 3 it explicitly says that engineers updated the statement in
response to early-bird participant queries, followed by:

- `Problem Statement last updated: 27 August 2026, 6:25PM`
- `Added Appendix: Test Shapes`
- `Updated torch_transformer_benchmark.py`

Canonical source:
<https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf#RNYvddBmXosHGbxr9jfmxYgOydd>.

## Attachment comparison

The revised PyTorch attachment has the same displayed size and byte count as
the early-bird file, but a different checksum. The complete source diff is
exactly two CLI-default changes:

```diff
-    parser.add_argument("--rtol", type=float, default=0.01)
-    parser.add_argument("--atol", type=float, default=0.001)
+    parser.add_argument("--rtol", type=float, default=0.02)
+    parser.add_argument("--atol", type=float, default=0.002)
```

| Artifact | Snapshot | Bytes | SHA-256 | Result |
|---|---|---:|---|---|
| PyTorch | 2026-08-26 early bird | 25,017 | `1bd12523657f338c09b53f0bb9052d9d16f728a71bd22bc8298567e1a4d78c22` | Archived unchanged |
| PyTorch | 2026-08-27 public update | 25,017 | `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e` | Current official file |
| TensorFlow | Re-downloaded 2026-08-28 | 54,531 | `00e99b6e1d19e961039b66eb3d3c055b36cc50f0436da2558f5f1fbe292ef798` | Byte-identical to 2026-08-26 |

One inconsistency remains inside the revised PyTorch file: its opening
docstring still says the old `atol=0.001` and `rtol=0.01 (1%)`, while its CLI
now defaults to `0.002` and `0.02`. Its comparison code still accepts equality
on both branches (`<=`), whereas the statement uses strict `<` wording.
TensorFlow still uses strict `<` for its absolute branch and `<=` for its
relative branch.

### Revised-file smoke verification

The following tiny, unmodified CPU command was run only to verify that the
revised attachment remains runnable and that its new defaults are active:

```bash
/usr/bin/time -l .venv/bin/python official/torch_transformer_benchmark.py \
  --batch-size 1 --seq-len 4 --d-model 8 --heads 2 --ffn-dim 8 \
  --layers 1 --device cpu --dtype float32 --accuracy-trials 1 \
  --warmup 1 --repeats 2 --benchmark-rounds 1
```

Observed exit code: `0`. The banner reported `abs_error <= 0.002 OR
relative_error <= 2.00%`; correctness passed with zero failures out of 32
elements. `/usr/bin/time -l` recorded 0.88 s real time, 201,949,184 bytes
maximum resident set size, 141,771,352 bytes peak footprint, and zero swaps.
The two-repeat CPU timing is deliberately not treated as a stable performance
result or evidence about any published shape.

## Newly published test shapes

The new section 3.7 appendix publishes these 14 rows. `TRUE` is shown in the
official table for every causal value.

| # | Batch | QKV dim | Heads | Sequence | Layers | Causal | FFN dim |
|---:|---:|---:|---:|---:|---:|:---:|---:|
| 1 | 64 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 2 | 1 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 3 | 4 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 4 | 16 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 5 | 128 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 6 | 10000 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 7 | 64 | 32 | 4 | 128 | 4 | TRUE | 32 |
| 8 | 64 | 1024 | 4 | 128 | 4 | TRUE | 1024 |
| 9 | 64 | 128 | 1 | 128 | 4 | TRUE | 128 |
| 10 | 64 | 128 | 2 | 128 | 4 | TRUE | 128 |
| 11 | 64 | 128 | 16 | 128 | 4 | TRUE | 128 |
| 12 | 64 | 128 | 4 | 32 | 4 | TRUE | 128 |
| 13 | 64 | 128 | 4 | 1024 | 4 | TRUE | 128 |
| 14 | 32 | 1024 | 16 | 100000 | 2 | TRUE | 1024 |

This is a substantial clarification of the previously unknown shape matrix,
but neither attachment's no-argument behavior executes this table:

- PyTorch still runs one default row: `B=8, S=128, D=512, H=8, F=2048,
  L=6`, non-causal.
- TensorFlow still generates 12 compact cases centered on `B=16, S=32,
  D=128, H=4, F=4D, L=6`, non-causal.
- The appendix instead uses `F=D`, `L=4`, and causal attention for rows 1–13,
  then `L=2` for row 14. TensorFlow's `--layers`, `--ffn-dim`, and `--causal`
  controls apply globally within one invocation, so one default invocation
  cannot express both appendix layer counts.

No official commands, dtype, padding configuration, framework-normalization
rule, or revised TensorFlow runner accompanied the appendix. Those execution
details therefore remain unresolved rather than being inferred.

The published stress row contains 5,120,000,000,000 attention-score elements.
The raw score tensor alone is 9.313 TiB at two bytes per value or 18.626 TiB at
four bytes per value. Using the reference-workload formula, the row has
12,605,440 trainable parameters and 1,350,985,318,400,000 counted matrix MACs.
It was not run locally.

## Clarified versus still unresolved

Now clarified:

- the exact 14-row shape table;
- causal attention is `TRUE` in every published row;
- per-row QKV dimension, heads, sequence length, layers, and FFN dimension;
- the current PyTorch CLI tolerance magnitudes now match the prose and
  TensorFlow defaults.

Still unresolved:

- the “given GPU model,” driver/toolkit, and framework versions;
- dtype and padding/mask configuration for the published rows;
- the official command sequence and how results across rows/frameworks are
  aggregated;
- whether any skip/OOM is disqualifying;
- strict-versus-inclusive correctness boundaries and the stale PyTorch
  docstring;
- redistribution licence for either organizer attachment.

Devpost Updates still contains no announcement, Discussions still reports no
topics, and the relevant binding Rules language remains unchanged as checked
on 2026-08-28. The Track 3 workshop was live during this audit; it was not
joined, and no verbal statement is represented here as an official
clarification.
