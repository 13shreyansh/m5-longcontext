# Milestone 150 — provisional champion manifest locked

> **Historical checkpoint — superseded for current status.** This file
> preserves the evidence and decisions available at its timestamp. Its
> pending/provisional, battery/power and then-current test-count wording is not
> the submission status. Use the current `README.md`, `TECHNICAL_REPORT.md` and
> `docs/CHAMPION_MANIFEST.json` for current claims.

Timestamp: **2026-08-30 14:55 SGT**

## Outcome

The current result now has a machine-readable provisional champion manifest at
`docs/CHAMPION_MANIFEST.json`. It identifies production-code commit `622c47c`,
clean-export evidence commit `5ec7121`, the complete rows-1–13 score evidence,
the selected BK48 attention and BM32/BN512 QKV geometries, the four direct-QKV
session results, the preceding exact-row observations, pinned upstreams,
verification commands and publication state.

The manifest deliberately reports the current route's exact B32 state as
`pending_stable_power_protocol` and stores the official MFU/combined score as
JSON `null`. It therefore cannot silently convert the preceding 98.589-second
point into a new-route observation or fabricate an organizer score.

## Source lock

Six production inputs are SHA-256 locked:

| File | SHA-256 |
|---|---|
| `solution/optimized_transformer.py` | `2bc156d2733206127ad9ee1ad2d645063fe51f572c7d08d6c283114f37999540` |
| `solution/mlx_nax_qkv_runtime.py` | `bc1ef3b32ea071c343886422f6df779a8d8d45f79d975647ce1b0e7890295059` |
| `solution/mps_metal4_qkv_head_layout.mm` | `b4586389f89155cbd431b60c02780d8fea290cde4fed26b1f3658a946da519fa` |
| `solution/mlx_nax_runtime.py` | `8b702f5051509f19e6981008057785b4f24123d0c2b4f539cc3661d6d4d9cb6e` |
| `solution/mps_metal4_attention.mm` | `7a4dbc10af20010e5895746f6ef2124e5c2b8e1be142304ca5db2a309f1e5a02` |
| `requirements-solution.txt` | `6304bea39686ea0956f835f96a7dad462426e450b0965355bd9e7b388214333e` |

The verifier fails if any file drifts, the manifest overstates the exact-row
state, the promoted tile changes, ordinary correctness is incomplete, or an
official score becomes non-null without an explicit manifest update.

## Exact commands and result

```bash
.venv/bin/python scripts/verify_champion_manifest.py
PYTHONPATH=. .venv/bin/python -m pytest -q
git diff --check
```

Observed output:

```text
champion manifest: OK (6 source hashes, stable-power exact row pending)
73 passed in 17.22s
```

All commands exited zero. This CPU/small-MPS verification does not produce a
performance point and no timing is added to the scorecard.

## Whole-suite scorecard

| Published rows | Current evidence | Change |
|---|---|---|
| 1–13 | organizer-default `high` points `2.513080x`/`2.523037x`/`2.508106x`, mean `2.514741x`; 117/117 timed-session comparisons plus 13/13 clean-export checks | evidence and source identity now machine-readable; performance unchanged |
| 14 | safe BK48 plus BM32/BN512 direct-head QKV; `1.019126x` balanced promotion; preceding 98.589-second exact point kept separate | manifest enforces the stable-power exact-B32 blocker; performance unchanged |

No official MFU or combined organizer score is inferred. This is a provisional
convergence control, not the final champion freeze. No external action occurred.
