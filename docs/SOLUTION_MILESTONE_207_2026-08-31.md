# Solution milestone 207 — final contention-controlled row-14 measurement

> **Historical checkpoint — current timing, superseded test count.** The three
> row-14 timings below remain canonical. The later `110/110` suite count in this
> file records that moment and is superseded by the current `111/111` suite.
> Use the current `README.md`, `TECHNICAL_REPORT.md` and
> `docs/CHAMPION_MANIFEST.json` for current status.

Snapshot: **2026-08-31 19:08 SGT**

## Outcome

The exact promoted-route protocol from clean commit `637b877` completed all
three B32/S100000 runs after an independent 60-sample clean window before each
run and five-minute cooldowns between runs:

| Run | Elapsed | Finite | Runtime contenders | Post-boundary failures |
|---:|---:|---:|---:|---:|
| 1 | `54.269253125 s` | yes | 0 | 0 |
| 2 | `51.895878706 s` | yes | 0 | 0 |
| 3 | `52.002793707 s` | yes | 0 | 0 |

The median is **52.002793707 seconds**, arithmetic mean
`52.722641846 seconds`, geometric mean `52.711389350 seconds`, range
`51.895878706–54.269253125 seconds`, and max/min ratio **1.045733389**. All
three outputs had exact shape `(32, 100000, 1024)`, float32 dtype, entirely
finite values and 32 item timings. No timing was trimmed. Power state was not
inspected or used as a gate.

The authoritative ignored evidence directory is
`artifacts/final-measurement/20260831T184501+0800/`. Summary SHA-256:
`c4d7aca767adee87a8a2dd70c6a82b4d080e7fe0b4109388b4fa0c7e0613d91c`.
Run-log SHA-256 values:

1. `ea999168b0a2c634c302591419642be7b9438ea6ebd766cdf9fcbe376de592cc`
2. `36b1ea6dfc527371857a3b48d5963b3841b937f10168ac14414f4256953f75b7`
3. `4416411d8e9e6a5f215c71320388a05ef73dfe64f88f8166eb4ef1cbf7f2851c`

`prepare_final_claims.py` independently revalidated the summary and raw logs.
The claim dry-run passed, then `apply_final_claims.py --write` atomically
updated the manifest, local drafts, storyboard, SVG cards, captions, public
README and readiness snapshot. A required second dry-run exposed a one-shot
applicator assumption; the applicator is now idempotent for the same exact
validated result while still rejecting drift. Ten focused tests pass.

The complete private suite then passed **110/110**. A disposable sanitized-tree
rehearsal passed **33 tests with one expected missing-organizer-input skip**;
after copying only the locally authorized organizer attachment and verifying
its source and destination SHA-256 as
`5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`,
the same tree passed **101/101**. The packaged manifest verifier passed in both
modes. This rehearsal confirms the current allowlist and counts; the two
byte-identical final release trees are still to be built from the exact commit.

No official MFU or combined organizer score is inferred. The preceding-route
`98.588985210`-second organizer-default point remains separately labelled.

## Whole-suite scorecard

| Published rows | Current verified evidence | Change in this milestone |
|---|---|---|
| 1–13 | Organizer-default `high` points `2.513080x`, `2.523037x`, `2.508106x`; arithmetic mean `2.514741x`, geometric mean `2.514733x`, spread `0.595334%`; 117/117 fresh float32 comparisons and the preceding all-dtype matrix pass | Performance unchanged |
| 14 | Safe BK48 attention plus BM32/BN512/BK256 direct-head QKV; four balanced B1 ratios `1.011633x`, `1.021096x`, `1.021071x`, `1.022742x`; geometric mean `1.019126x`, 33/36 positive; current-route exact B32 runs `54.269253125/51.895878706/52.002793707 s`, median `52.002793707 s`, max/min `1.045733389`; all outputs finite and zero monitored contenders | Final local contention-controlled timing gate completed |

No publication, submission, push, organizer contact, registration change or
repository visibility change was performed.
