# AI and tool-use disclosure — 2026-08-29 SGT

This is a truthful internal disclosure record for the Track 3 work in this
repository. It is not a Devpost submission, does not expose private message
content or credentials, and must be refreshed at the final action boundary.

## Roles and human steering

- **Human entrant:** selected Track 3, supplied the official webinar transcript
  and screenshots, directed the task toward the participant's own Apple
  machine, asked that relevant open-source work be read and experimentally
  compared, required repeated introspection, and authorized challenge-window
  experimentation. The repository does not infer a complete team roster or
  contribution split.
- **Codex AI coding agent:** read and reconciled official material; acquired and
  checksum-preserved the organizer attachments; inventoried the machine;
  researched and pinned upstream sources; generated and edited experiment,
  solution, test and documentation files; executed local correctness,
  performance, memory and provenance checks; rejected failing/noisy variants;
  and prepared this report. The human steered the objective and corrected the
  strategic assumption that CUDA was necessary; the agent executed the
  recorded local workflow.
- **Child agents:** none were used inside this Track 3 task. Work in other
  isolated TechJam track tasks is outside this repository and is not represented
  as Track 3 work.

The current Codex model label was not independently recorded in a durable,
user-visible receipt, so this document does not guess it. The exact model label
shown by the host at submission time should be inserted before publication.

## AI/tool capabilities used

- signed-in Chrome/Lark browsing for the official statement and attachments;
- web/source research for primary documentation and relevant open-source
  repositories;
- local shell and Git for isolated environments, build/test commands, source
  inspection, checksums, resource probes and local commits;
- Python with PyTorch/MPS for reference comparisons, benchmarking and tests;
- Metal shader compilation through `torch.mps.compile_shader` plus a narrow
  Objective-C++ Metal-4 bridge for the retained Apple GPU kernels;
- MLX and Swift build/test probes for upstream feasibility; and
- structured experiment, provenance and report generation in this repository.
- local inspection of the supplied webinar transcript and screenshots used to
  reconcile the spoken guidance with the written challenge materials.

No Outlook password, Lark password, API key, private email content or other
secret is stored in source, documentation, Git history or this disclosure.
No ElevenLabs request was made, and the credential exposed in chat was not
used, stored, logged or copied into this repository.

## Material third-party sources

The retained solution materially derives Metal code from the MIT-licensed
`triton-msl` repository at commit
`182c1820fd24a836d565e1da842f28414de64084` and materially reuses Apple's
MIT-licensed MLX NAX attention headers from commit
`3f0bd54ff0c0af5b88530191d5df31010ce54fcd`. The retained MLX template is
locally instantiated as BQ256/BK48/BD64/WM16/WN1 after correctness-first query-
and key-tile sweeps. Both exact licence notices are committed; the executable verifier
regenerates the triton-msl shaders, byte-matches all eight MLX headers, and
asserts the tuned template instantiation. Philip Turner's `metal-flash-attention`,
`mps-flash-attention`, `mlx-flashattention-steel`, and the `mlx.fast` Gemma
engine were inspected or tested as research sources but did not enter the
retained solution wholesale. `mlx-train-perf` at MIT commit `cdfce970` was also
acquired and source-audited after its release; its training-memory attention
path was closed without build and contributed no solution code.
The Apache-2.0 `mlx-metal-kernels` checkout at `9fc1d38a` was also acquired
into ignored storage and closed by source audit; no dependency, kernel or code
from it entered the retained solution.
Milestone 148 additionally retains the same pinned MLX NAX linear template as
a BM32/BN512/BK256/WM1/WN8 packed-QKV projection with entrant-added direct
head-major storage, a narrow fail-closed PyTorch MPS bridge and a prior-route
fallback. The existing Apple MIT notice covers the reused headers; the
provenance verifier asserts the generated geometry and store transformation.
The exact used/tested/rejected mapping, pins and licence status are in
[`SOURCE_TO_EXPERIMENT_LEDGER_2026-08-29.md`](SOURCE_TO_EXPERIMENT_LEDGER_2026-08-29.md)
and [`UPSTREAM_EXPERIMENT_AUDIT_2026-08-29.md`](UPSTREAM_EXPERIMENT_AUDIT_2026-08-29.md).

## Interaction and accounting snapshot

The complete human/AI interaction history resides in the private Codex task and
should be exported or linked only through an organizer-approved, privacy-safe
mechanism. It is not copied into this repository because it may contain private
account context and because a transcript can be much larger than the source.

### Current pre-submission accounting checkpoint

At 2026-09-01 00:58 SGT, after the milestone-252 judge-facing evidence
reconciliation, the still-active Codex goal reported this cumulative host
goal-accounting snapshot:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 20,090,560 |
| Goal-accounting elapsed time | 155,931 seconds (43 h 18 m 51 s) |
| Child agents used in this Track 3 task | 0 |
| Exact host-visible model label | Unavailable; not guessed |
| Private repository test scope | 111/111 passed |
| Public-safe release test scope | 33 passed, 1 expected missing-input skip |
| Authorized-input release test scope | 101/101 passed |
| Latest verified release source | `9c4f555d8101b3a9a41c9b3c6cbb58568ed96cce` |
| Latest neutral no-remote handoff | `0e3782a32bcda0c19b8940ee7d947fc022a7f8c4` |

This table supersedes every historical checkpoint below. The goal remains
active, so it is a pre-submission checkpoint, not a final submission-time
receipt. The values are cumulative and must not be added to earlier snapshots.
A final action-time refresh is still required after all work stops.

### Historical cumulative checkpoints

At the live milestone-142 checkpoint on 2026-08-30, the still-active Codex goal
reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 11,534,617 |
| Goal-accounting elapsed time | 61,135 seconds (16 h 58 m 55 s) |
| Child agents used in this Track 3 task | 0 |

These are cumulative host goal-accounting values, not a provider invoice, not
hardware-kernel time, and not final Devpost-submission totals. They include
research, reasoning, tool orchestration, documentation and local command work
in this task. A final snapshot must replace or append to this table immediately
before submission; do not sum repeated cumulative snapshots.

At the live milestone-143 checkpoint on 2026-08-30, after the balanced
direct-head QKV rejection and restored-champion regression, the same cumulative
goal counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,005,684 |
| Goal-accounting elapsed time | 62,352 seconds (17 h 19 m 12 s) |
| Child agents used in this Track 3 task | 0 |

This later table supersedes the milestone-142 snapshot for current accounting;
the two cumulative values must not be added.

At the live milestone-144 checkpoint, after the layer-major schedule was
rejected at bounded scale, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,067,142 |
| Goal-accounting elapsed time | 62,674 seconds (17 h 24 m 34 s) |
| Child agents used in this Track 3 task | 0 |

This table supersedes both earlier checkpoint snapshots; cumulative snapshots
must not be summed.

At the live milestone-145 checkpoint, after the exact Metal grid permutation
was rejected at bounded scale, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,109,834 |
| Goal-accounting elapsed time | 62,864 seconds (17 h 27 m 44 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes the earlier checkpoint snapshots; none of the
cumulative values should be summed.

At the live milestone-146 checkpoint, after direct-head QKV Metal math modes
were rejected, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,153,882 |
| Goal-accounting elapsed time | 63,080 seconds (17 h 31 m 20 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-147 checkpoint, after completing all eight direct-head
QKV tile families, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,191,730 |
| Goal-accounting elapsed time | 63,350 seconds (17 h 35 m 50 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-148 checkpoint, after the BM32/BN512 direct-head QKV
promotion and production regression, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,383,698 |
| Goal-accounting elapsed time | 64,414 seconds (17 h 53 m 34 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-149 checkpoint, after the promoted-champion clean-export
audit, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,482,048 |
| Goal-accounting elapsed time | 64,996 seconds (18 h 3 m 16 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-150 checkpoint, after adding the provisional champion
manifest and drift verifier, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,528,356 |
| Goal-accounting elapsed time | 65,265 seconds (18 h 7 m 45 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-151 checkpoint, after building and independently
verifying the deterministic sanitized release, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,705,160 |
| Goal-accounting elapsed time | 66,960 seconds (18 h 36 m 0 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-152 checkpoint, after the fresh sanitized-release
dependency, native-build and full-test rehearsal, the cumulative counter
reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,739,481 |
| Goal-accounting elapsed time | 67,235 seconds (18 h 40 m 35 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-153 checkpoint, after the targeted public-source refresh
and exact latest-tag reconciliation, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,794,758 |
| Goal-accounting elapsed time | 67,404 seconds (18 h 43 m 24 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-154 checkpoint, after the verified three-minute
storyboard and sanitized-package refresh, the cumulative counter reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 12,843,872 |
| Goal-accounting elapsed time | 67,702 seconds (18 h 48 m 22 s) |
| Child agents used in this Track 3 task | 0 |

This latest table supersedes all earlier checkpoint snapshots; cumulative
values must not be summed.

At the live milestone-161 checkpoint, after technical convergence, exact-shape
promoted-route completion, final-claim validation, corrected dual-mode package
rehearsal and the authenticated Devpost refresh, the cumulative counter
reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 13,420,511 |
| Goal-accounting elapsed time | 70,679 seconds (19 h 37 m 59 s) |
| Child agents used in this Track 3 task | 0 |

This milestone-161 table supersedes all earlier checkpoint snapshots. It is
still not the final submission-time total and must be replaced or explicitly
superseded at the final action boundary; cumulative snapshots must not be
summed.

At the live milestone-180 checkpoint, after the contention-controlled final
protocol hardening, rejected-run evidence, public-report reconciliation and
current 107-test/provenance/claim verification, the cumulative counter
reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 15,690,449 |
| Goal-accounting elapsed time | 120,104 seconds (33 h 21 m 44 s) |
| Child agents used in this Track 3 task | 0 |

This milestone-180 table supersedes all earlier checkpoint snapshots. It is
still not the final submission-time total; cumulative snapshots must not be
summed.

At milestone 196, the regression suite advanced to 108 tests after adding
coverage for generalized Python/Node/Codex contention detection and
irrecoverable partial-spread rejection. The goal-accounting values above were
not refreshed at that code-only checkpoint and remain explicitly dated.

At the milestone-210 final local package and fresh-environment checkpoint, the
still-active Codex goal reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 17,753,439 |
| Goal-accounting elapsed time | 136,989 seconds (38 h 3 m 9 s) |
| Child agents used in this Track 3 task | 0 |

This milestone-210 table supersedes all earlier checkpoint snapshots. The goal
remains active, so it is not represented as a final submission-time receipt;
cumulative snapshots must not be summed. The host-visible exact model label
remains unverified and is not guessed.

At the milestone-220 six-card package and clean-handoff checkpoint, the same
still-active goal reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 18,455,323 |
| Goal-accounting elapsed time | 141,975 seconds (39 h 26 m 15 s) |
| Child agents used in this Track 3 task | 0 |
| Private repository test scope | 111/111 passed |
| Public-safe release test scope | 33 passed, 1 expected missing-input skip |
| Authorized-input release test scope | 101/101 passed |

This milestone-220 table supersedes the earlier accounting snapshots. The goal
remains active, so these are not final submission-time totals and must not be
summed. The exact verified release content commit is carried by the enclosing
package's `RELEASE_METADATA.json`; the organizer attachment remains outside
its signed manifest. The host-visible exact model label remains unavailable in
the observed Codex Desktop/thread metadata and is therefore recorded as
unavailable rather than guessed.

At the milestone-231 privacy-hardened package and clean-handoff checkpoint,
the same still-active goal reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 18,787,010 |
| Goal-accounting elapsed time | 145,125 seconds (40 h 18 m 45 s) |
| Child agents used in this Track 3 task | 0 |
| Private repository test scope | 111/111 passed |
| Public-safe release test scope | 33 passed, 1 expected missing-input skip |
| Authorized-input release test scope | 101/101 passed |

This milestone-231 table supersedes all earlier accounting snapshots. The goal
remains active, so it is not a final submission-time total and the cumulative
tables must not be summed. The content release and clean local handoff are
identified in milestones 230 and 231. The host-visible exact model label
remains unavailable and is not guessed.

At the milestone-246 disclosure-refresh checkpoint, after the literal README
repair, corrected deterministic package, clean one-commit handoff, complete
recipient workflow and post-setup verifier boundary audit, the same still-active
goal reported:

| Item | Snapshot value |
|---|---:|
| Goal-accounting tokens used | 19,309,262 |
| Goal-accounting elapsed time | 149,971 seconds (41 h 39 m 31 s) |
| Child agents used in this Track 3 task | 0 |
| Private repository test scope | 111/111 passed |
| Public-safe release test scope | 33 passed, 1 expected missing-input skip |
| Authorized-input release test scope | 101/101 passed |

At that checkpoint, the milestone-246 table superseded the earlier accounting
snapshots. It was not a final submission-time receipt; cumulative snapshots
must not be summed. Its then-current verified release source was `c93879b`, and
its neutral no-remote handoff was `3e40cfd`. The host-visible exact Codex model
label remained unavailable and was not guessed.

## Human-intervention summary

Material steering visible in the task included:

1. preserve the organizer resources and work only in the isolated Track 3
   repository;
2. re-check the statement after public release and after the official start;
3. treat the user's own Apple machine as the result target after workshop
   clarification, rather than assuming CUDA was required;
4. investigate MLX, `mlx.fast`, Philip Turner and other open-source work as
   research sources, while testing rather than assuming their suitability; and
5. continue for the challenge window with quantified goals, recurring reviews
   and evidence-based status reporting; and
6. make the final exact-row measurement contention-controlled without querying
   or using charge state; and
7. reject the existing local video draft, pause video generation, and use
   ElevenLabs narration only after resumption with a newly rotated credential.

This summary describes steering, not hidden reasoning. Exact prompts and agent
responses should be supplied only if the submission process explicitly asks for
interaction history and provides an appropriate privacy boundary.

## Publication checklist

Before any public release or submission:

- record the final cumulative token/time snapshot and any later AI roles; if
  the exact model label remains unavailable, retain that explicit boundary;
- add the verified entrant/team contribution split;
- remove organizer attachments unless redistribution permission is verified;
- rerun checksums, provenance verification and the complete test suite from the
  final commit;
- make sure every copied/adapted component retains its licence and attribution;
- retain the rejected-video and human-approval gates until a complete
  replacement has been explicitly reviewed; and
- separately authorize any repository visibility change, remote push, video
  publication or Devpost submission.
