# Manifest-backed video evidence cards

Status: **claim-synchronized source assets only; prior assembled draft rejected;
video generation paused by user**. Do not run a media builder until explicit
resumption. On resumption, narration must use ElevenLabs with a newly rotated
runtime-only credential that never enters source, logs or Git history, and the
complete replacement must receive human approval.

Six 1920×1080 SVG cards support the final local three-minute storyboard:

1. `video_assets/00_hook.svg` contrasts the 18.626-TiB explicit tensor with the
   complete bounded run;
2. `video_assets/01_architecture.svg` explains the bounded row-14 path;
3. `video_assets/02_results.svg` shows rows 1–13 and row-14 claims with their
   distinct evidence boundaries; and
4. `video_assets/04_row14_evidence.svg` shows the three exact runs and their
   control boundary;
5. `video_assets/05_reproducibility.svg` shows source, package and privacy
   gates; and
6. `video_assets/03_boundaries.svg` keeps proven facts and limitations visible
   in the closing frame.

They are generated only from `docs/CHAMPION_MANIFEST.json` plus fixed visual
copy. Rebuild and verify them with:

```bash
.venv/bin/python scripts/build_video_assets.py --write
.venv/bin/python scripts/build_video_assets.py
```

The default command byte-compares every committed card against freshly
rendered content and fails if a manifest-backed value or the final row-14
boundary drifts. The cards contain no organizer attachment, credential, raw
private log, official MFU or combined organizer score.

`video_assets/narration.srt` is generated from the same verified storyboard.
Sentence-level cue lengths are allocated by word count inside each fixed
storyboard section, so captions cover the complete 178-second timeline without
manually copying claims:

```bash
.venv/bin/python scripts/build_video_captions.py --write
.venv/bin/python scripts/build_video_captions.py
```

These cards are local recording assets and are included in the sanitized
release after the final row-14 protocol updated the champion manifest and the
cards were regenerated and rechecked.

## Historical local draft builder — generation blocked

The prior 178-second synthetic-voice draft was technically valid but rejected
as a submission candidate. Its macOS `say`/Swift/AppKit/AVFoundation builder is
retained only as historical timing and media-audit evidence. While
`video_generation_paused_by_user=true` in `docs/FINAL_READINESS.json`, the
Python entry point fails before creating an output directory:

```bash
.venv/bin/python scripts/build_local_video_draft.py
```

After a future explicit resumption and replacement of the narration path, the
non-overwriting builder writes only under ignored `artifacts/`, uses a
195-words-per-minute default that fits the current final narration, checks the
conservative 180-second limit, rejects narration that would be truncated,
verifies one 1920x1080 H.264 video track plus one narration track, and records
durations and hashes of every source and output. The voice is synthetic and
the result is explicitly unpublished. It is a fallback proof and timing
rehearsal, not a replacement for a final human-reviewed demo or an
authorization to publish. The system encoder may emit different MP4 bytes on
repeat runs, so byte-identical video is not claimed; each output manifest locks
that output while the source/audio hashes, duration, dimensions and track
counts provide the repeatable contract.

Audit a completed local draft with:

```bash
.venv/bin/python scripts/audit_local_video_draft.py \
  artifacts/local-video-draft-TIMESTAMP
```

The audit rechecks the manifest hashes and byte count, uses AVFoundation to
decode and verify duration, dimensions and track count, cross-checks indexed
codec metadata when available, decodes the AAC track actually embedded in the
MP4, rejects excessive clipping or more than three whole silent seconds, and
samples the complete timeline every two seconds plus both sides of every card
boundary. Blank/solid sampled frames fail closed. This automated gate does not
replace a complete human watch with audio.
