# Three-minute Track 3 video storyboard

Status: **numerical claims synchronized to the final contention-controlled
measurement; prior assembled draft rejected; video generation paused by user**.
Do not record or assemble media until explicit resumption. After resumption,
use ElevenLabs narration with a newly rotated runtime-only credential and
require a complete human watch before treating the replacement as usable.

The binding-safe target is at most three minutes. The machine-readable source
is `docs/VIDEO_STORYBOARD.json`; `scripts/verify_video_storyboard.py` checks the
178-second timeline and locks every numerical claim to
`docs/CHAMPION_MANIFEST.json`.

## Shot plan

| Time | Purpose | Screen evidence |
|---:|---|---|
| 0:00–0:12 | Hook with the physically impossible explicit row-14 tensor | Published shape and `18.626 TiB` calculation |
| 0:12–0:30 | Define the machine and correctness contract | M5 Pro inventory, 14 rows, `abs <= 0.002 OR rel <= 0.02` |
| 0:30–0:58 | Show the complete rows-1–13 result | Three `high` aggregate points, 2.514741x mean, 117/117 pass |
| 0:58–1:44 | Explain the bounded architecture | Direct-head QKV, BQ256/BK48 NAX attention, fp32 boundary and fallback |
| 1:44–2:17 | Show row-14 evidence without overclaiming | S8192 reference, four balanced QKV sessions, exact-run status |
| 2:17–2:43 | Prove reproducibility and provenance | Six hashes, public-safe 33-pass/1-skip gate, authorized-input 101-pass gate, clean release, MIT notices |
| 2:43–2:58 | State limitations and close | No official MFU, no full S100000 reference, device/control boundary |

## Required recording commands

Run these from the final committed tree immediately before recording:

```bash
.venv/bin/python scripts/verify_champion_manifest.py
PYTHONPATH=. .venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_video_storyboard.py
```

For the package shot, run the verifier from the eventual final sanitized tree:

```bash
python3 scripts/verify_release_manifest.py
```

For the hardware shot, show the exact output rather than typing specifications
onto a slide. Do not inspect or display battery, charging, thermal, or power
state:

```bash
system_profiler SPHardwareDataType SPDisplaysDataType
sw_vers
```

## Row-14 recording gate

Current promoted-route exact B32 status:

```text
COMPLETE_CONTENTION_CONTROLLED_PROTOCOL
RUNS_SECONDS=54.269, 51.896, 52.003
MEDIAN_SECONDS=52.003
RANGE_SECONDS=51.896–54.269
ALL_OUTPUTS_FINITE=True
POWER_STATE_CHECKED=False
```

The existing `98.588985210`-second organizer-default point belongs to the
**preceding** safe projection route. It may be shown only with that label. Do
not call it the promoted direct-head QKV result. The final performance segment
must show the new three-run median and range from the signed manifest.

## Editing constraints

- Use terminal recordings or static evidence captured from the final commit;
  do not animate invented benchmark output.
- Keep every number visible long enough to read, with the comparison boundary
  in the same frame.
- Do not show private Lark/Outlook content, organizer attachments, credentials,
  raw private Git history, or ignored caches.
- Do not call local throughput an official MFU or combined organizer score.
- Show the public-safe repository structure, not this private repository.
- Add captions because much of the evidence consists of exact numbers and
  commands.
