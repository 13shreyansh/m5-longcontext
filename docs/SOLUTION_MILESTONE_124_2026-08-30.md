# Milestone 124 — live release-contract refresh

Timestamp: **2026-08-30 07:58 SGT**

## Outcome

The live official surfaces show no post-start Track 3 statement change or
clarification. The signed-in Lark information document reports `Last modified:
11:23 PM Aug 28`; its Track 3 section still says the problem statement was last
updated on **2026-08-27 18:25 SGT**, when the test-shape appendix was added and
`torch_transformer_benchmark.py` was updated. Devpost Updates contains only the
empty announcement placeholder, and Discussions reports that no topics exist.

The release requirements are therefore unchanged and concrete:

1. an English Devpost project description naming the tools, APIs, libraries,
   assets and working result;
2. a **public code repository** with structured source and a README covering
   setup, reproduction, limitations and contributions;
3. a **public YouTube demo video**, conservatively held to three minutes or
   less, linked from Devpost; and
4. free, unrestricted judge access to a functioning project or test build
   through the judging period.

No Devpost project has yet been started on the signed-in account. The live
submission manager displayed `Start project` / `Create project` and no existing
hackathon project. This is a readiness gap, not a technical-result failure. No
button was clicked and no project, repository, video or submission was created,
published or changed during this read-only audit.

## Current result remains unchanged

| Published rows | Current evidence | Status |
|---|---|---|
| 1–13 | Three complete synchronized points `2.499365x`, `2.517673x`, `2.441356x`; arithmetic mean `2.486131x`, geometric mean `2.485918x`; fresh `117/117` trials over `39/39` case/dtype combinations; `65/65` tests | Correct and repeatable on the declared M5 Pro; not an official score |
| 14 | Three exact B32/S100000 runs `69.605358`, `97.172911`, `131.088197` seconds; median `97.172911` seconds; every `3,276,800,000`-element output finite; five paired BK48/BK32 sessions `1.082579x` geometric mean | Full explicit baseline infeasible because one score tensor is `18.626 TiB`; no official MFU claimed |

The production source is unchanged from clean-export champion `0deb070`; the
commits after it contain rejected experiments and evidence only. The live rule
audit does not justify changing the kernel.

## Release gap matrix

| Gate | Current state | Required closeout |
|---|---|---|
| Working source | Ready locally | Freeze a final commit after the convergence window |
| Correctness and tests | Ready at current champion | Re-run from the frozen commit in a fresh dependency environment |
| Technical report | Substantive draft ready | Condense final numbers and entrant/team details |
| AI/source disclosure | Draft ready | Replace cumulative accounting snapshot and add verified model/team facts |
| Public-safe repository | Not ready | Create a new sanitized history that excludes organizer attachments and private acquisition metadata |
| Demo video | Not created | Record a <=3-minute hardware/correctness/performance/provenance/limits walkthrough |
| Devpost project | Not started | Create and fill only after action-time authorization |

The current private repository cannot simply be made public: its Git history
contains organizer attachments without a visible redistribution licence. The
safe route remains a new, allowlisted repository containing only entrant source,
tests, pinned requirements, required licences, a public README, report and
privacy-safe disclosure.

## Exact inspected official surfaces

- Lark information document:
  <https://bytedance.larkoffice.com/wiki/GdYFwzWNLiREsSkuIjZcDznInWc>
- Devpost submission manager:
  <https://devpost.com/submit-to/30686-tiktok-techjam-2026/manage/submissions>
- Devpost Resources: <https://tiktoktechjam2026.devpost.com/resources>
- Devpost Updates: <https://tiktoktechjam2026.devpost.com/updates>
- Devpost Discussions:
  <https://tiktoktechjam2026.devpost.com/forum_topics>
- Binding Official Rules: <https://tiktoktechjam2026.devpost.com/rules>

## Time boundary

At the snapshot, **19 h 58 m 21 s** of the challenge had elapsed and
**52 h 01 m 38 s** remained. The agreed operating plan still applies: use the
pre-48-hour period for only high-value correctness-gated technical work, hours
48–60 for convergence and freeze, and hours 60–72 for the sanitized package,
video, final report/disclosure and submission readiness.
