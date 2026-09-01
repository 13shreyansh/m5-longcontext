# Track 3 final action handoff

Status: **verified local pre-action checklist — no external action authorized**

This file is an operator handoff, not permission to publish or submit. It is
the human-readable companion to `docs/FINAL_READINESS.json`. The signed
sanitized tree is the only publication source. Never change the existing
private repository's visibility, and never put the organizer attachment,
credentials, private history, caches or raw logs into a public repository.

The current validators are **pre-action guards**. They deliberately require
identity and URL fields to remain unset and publication booleans to remain
false. A green local gate is not a public URL, human video approval, or a
Devpost receipt.

## Current verified local state

- Rows 1–13: three organizer-default `high` sessions at `2.513080x`,
  `2.523037x` and `2.508106x`; arithmetic mean `2.514741x`; 117/117 fresh
  float32 comparisons pass.
- Row 14: bounded BK48 attention plus BM32/BN512/BK256 direct-head QKV; three
  exact promoted-route runs at `54.269253125`, `51.895878706` and
  `52.002793707` seconds; median `52.002793707`; every output finite.
- Official MFU or combined organizer score: unavailable and not inferred.
- Video: generation paused by user; the prior draft is rejected and has no
  approval or publication status.
- External state: no push, public-repository change, video publication or
  Devpost submission is represented by this package.

## The nine unresolved gates

Complete these gates in order. Unknown, intended, drafted, uploaded and
submitted are different states.

| Gate ID | Owner and required evidence | Safe transition |
|---|---|---|
| `project_name` | Entrant supplies the exact final English project name | Copy only the confirmed spelling into the Devpost draft/form |
| `entrant_team_attribution` | Entrant verifies solo/team status, display names, contributions and team representative if applicable | Do not infer identity from filenames, email, Git or chat history |
| `video_draft_human_approved` | Entrant watches the complete replacement with audio and explicitly approves it | Video generation remains paused until explicit resumption |
| `public_repository_url` | A public URL to the new sanitized one-commit history | Open it signed out; verify README, commit and manifest, and verify the organizer attachment is absent |
| `public_video_url` | A public YouTube URL for the approved replacement | Open it signed out; verify playback, audio, readability and duration |
| `repository_pushed` | Read-back proves the sanitized handoff commit reached the intended remote | A local commit or configured remote is not proof |
| `repository_public` | Signed-out access to the repository succeeds | Never make the private preparation history public |
| `video_published` | Signed-out playback of the final approved video succeeds | An upload draft or processing screen is not proof |
| `devpost_submitted` | Devpost shows a final receipt/status after submission | A filled form, preview or saved draft is not proof |

## Stage A — verify the signed sanitized tree

Run from the exact sanitized publication candidate after creating its local
`.venv` from `requirements-lock.txt`:

```bash
git status --short
.venv/bin/python scripts/verify_release_manifest.py
./scripts/acquire_solution_upstreams.sh
.venv/bin/python scripts/verify_solution_provenance.py
.venv/bin/python scripts/verify_champion_manifest.py
.venv/bin/python scripts/verify_devpost_draft.py
.venv/bin/python scripts/verify_video_storyboard.py
.venv/bin/python scripts/build_final_readiness.py
PYTHONPATH=. .venv/bin/python -m pytest -q -p no:cacheprovider
git diff --check
git fsck --full
```

Required observations before any action-time request:

- every command exits zero;
- `git status --short` is empty;
- the release verifier reports that the organizer attachment is outside the
  signed manifest;
- public-safe tests report 33 passing tests and one expected missing-input
  skip;
- the authorized-input copy, kept private and disposable, reports 101/101;
- `FINAL_READINESS.json` reports the same nine blockers listed above; and
- no official MFU or combined score appears.

The private release builder is deliberately absent from the public package.
Build and compare two non-overwriting release trees in the private workspace,
then create a new neutral one-commit history from the byte-identical tree. Do
not publish the existing private remote or any earlier candidate by accident.

## Stage B — identity and narration inputs

Before changing any public field, obtain these exact entrant-owned facts:

1. final project name;
2. solo entrant or complete team attribution and contributions;
3. team representative, if this is a team entry; and
4. explicit instruction that video work may resume.

If video work is resumed, use ElevenLabs only with a **newly rotated**
runtime-only credential supplied through an environment/secret mechanism. Do
not paste it into chat, a command argument, source, logs, captions, metadata or
Git. The previously exposed credential is not usable evidence of safe secret
handling.

The replacement video must remain at or below the conservative three-minute
limit and must be watched in full with audio. Automated codec, frame and
silence checks do not constitute human approval. Do not generate, upload or
publish media while `video_generation_paused_by_user` is true.

## Stage C — separately authorized external actions

External actions require a separate action-time authorization after the exact
target and payload are shown. The safe order is:

1. create or select the intended public repository without altering the
   private preparation repository;
2. push only the neutral sanitized one-commit history;
3. read back the public repository signed out and record its exact commit;
4. publish only the human-approved replacement video;
5. read back the public video signed out;
6. populate and preview the Devpost entry with verified identity, public URLs,
   environment, tools, AI assistance, licences, results and limitations;
7. submit only after a separate final authorization; and
8. read back and preserve the Devpost receipt/status.

The repository, video and Devpost fields must agree on the Apple M5 Pro result,
all 14 published rows, the infeasible explicit row-14 reference boundary,
third-party MIT provenance, AI assistance, and the absence of an official MFU
or combined score.

## Stage D — receipt audit

After authorized actions, preserve a private receipt record containing only:

- final sanitized source commit and release-manifest SHA-256;
- public repository URL and signed-out verification time;
- public video URL, duration and signed-out verification time;
- Devpost project URL, receipt/status and observed submission time; and
- exact entrant-confirmed project/team attribution.

Do not put credentials, private browser data, organizer attachments or raw
private logs into the receipt. Do not edit the already signed public package to
pretend that external actions occurred. Current local validators must pass
immediately before action; later receipt verification is a distinct audit of
observed external state.

## Deadline boundary

The public Devpost overview and Official Rules state the submission deadline
as **2026-09-01 12:00 SGT**. Stop substantive edits with enough time for
upload, public read-back, preview and receipt verification. An action started
before the deadline but not received by Devpost is not a verified submission.
