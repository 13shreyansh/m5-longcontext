# Track 3 submission and evidence contract

Snapshot: **2026-08-26 SGT**; latest public refresh: **2026-09-01 00:18 SGT**;
latest authenticated submission-manager refresh: **2026-08-30 21:39 SGT**

This is a preparation-only reconciliation of the official Track 3 statement,
the live Devpost overview, and the binding Official Rules. It is not a
submission draft, judged-solution plan, or authorization to publish anything.
The Official Rules state that they control if other TechJam materials are
inconsistent.

## Live post-start refresh

At 07:58 SGT, the signed-in official information document reported `Last
modified: 11:23 PM Aug 28`. Its Track 3 notice still identified the last
track-specific change as **2026-08-27 18:25 SGT**: the test-shape appendix was
added and the PyTorch attachment was updated.

At 21:39 SGT, the authenticated Devpost submission manager showed no existing
project, offered `Start project` / `Create project`, displayed `1 more day` and
the exact deadline **1 Sept 2026 @ 12:00pm GMT+8**. Updates still contained no
organizer announcement and Discussions still said no topics had been created.
The overview page's `Submissions open soon` text conflicts with the active
submission manager and is treated as stale display copy, not the operational
state. A new Lark tab required the document password; no password was entered
or transmitted during this refresh, so the 07:58 authenticated Lark content
snapshot remains the latest verified statement body. Nothing was created,
edited, published or submitted. See milestones 124 and 161.

At 13:00 SGT on August 31, a fresh public read-only refresh found no new
organizer announcement on Devpost Updates, no new resource beyond the same
information-document link, and no Discussion topic. The public Rules page
still states the deadline as **September 1, 2026 at 12:00pm SGT** and the
72-hour submission period as August 29 12:00pm through September 1 12:00pm
GMT+8. This public refresh did not re-authenticate the Lark document or the
submission manager and therefore does not supersede their separately labelled
authenticated snapshots. Nothing was created, edited, published or submitted.
See milestone 182.

At 14:22 SGT on August 31, another public read-only refresh again found no
announcement, Track 3 resource, Discussion topic or rules/deadline change.
Updates still contains only the organizer placeholder, Resources still points
to the same information document, Discussions still says no topics have been
created, and the Rules page still states September 1 at 12:00 SGT. The visible
participant count changed, but that is not a Track 3 clarification or scoring
change. No authenticated page or external state was opened or modified. See
milestone 186.

At 19:50 SGT on August 31, the final public read-only refresh again found no
announcement, new Track 3 resource, Discussion topic, rules change or deadline
change. The Rules page still requires a public repository with README, retains
the open-source licence-and-enhancement condition and four equally weighted
Stage Two criteria, and still shows September 1 at 12:00 SGT. A newly opened
Lark tab required the document password and was not re-authenticated, so this
refresh does not supersede the separately recorded authenticated statement
snapshot. No password or other data was transmitted and no external state was
modified. See milestone 212.

At 22:04 SGT on August 31, a final-hours public read-only refresh again found
no announcement, new resource, Discussion topic, deadline change or rules
change. The overview and Official Rules still show **September 1, 2026 at
12:00pm SGT**; Updates still contains only the organizer placeholder;
Resources still points only to the same information document; and Discussions
still says no topics have been created. The overview still requires a written
description, public code repository with README and public three-minute
YouTube demonstration. Participant counters differed across independently
crawled public pages and are treated as display/cache noise, not a Track 3
clarification. No authenticated page was opened and no external state was
modified. See milestone 234.

At 00:18 SGT on September 1, a new public read-only refresh again found no
announcement, new resource, Discussion topic, deadline change or rules change.
The overview and Official Rules still state **September 1, 2026 at 12:00pm
SGT**. Updates still contains only the organizer placeholder, Resources still
points only to the same information document, and Discussions still reports
that no topics have been created. Independently crawled public pages showed
different participant counts (`3129`–`3135`); that remains display/cache noise,
not a Track 3 clarification. The overview still requires a written
description, public repository with README and public three-minute YouTube
demo. No authenticated page was opened and no external state was modified.
See milestone 250.

## Time and creation boundary

- The 72-hour Submission Period is **2026-08-29 12:00 SGT through
  2026-09-01 12:00 SGT**. Judging starts at 15:00 SGT on September 1 and ends
  at 15:00 SGT on September 7.
- A project must be newly created by the entrant or, if it existed before the
  Submission Period, be **significantly updated after the Submission Period
  starts**. This preparation repository predates the window, so none of its
  pre-start work is represented as the judged project or as satisfying that
  requirement.
- Draft submission fields may be edited until the deadline. Afterward, changes
  are prohibited unless the Sponsor and Devpost specifically permit a narrow
  change to remove or replace possible infringement, personally identifiable
  information, or inappropriate material; the submission must remain
  substantively the same.

Source: <https://tiktoktechjam2026.devpost.com/rules>, sections 1, 4, and 5.

## Required submission package

The sources collectively require all of the following in English:

1. A Devpost text description explaining the working project, its features,
   and its technical stack. The rules expressly call for development tools,
   APIs, assets, and libraries where applicable. The Track 3 statement also
   asks for environment details (CPU, GPU, and disk), optimization description,
   final results, and AI skills/tools used.
2. A **public code repository** with commented, well-structured code and a
   comprehensive README. The Track 3 statement asks that the README cover the
   overview, setup and reproduction, limitations/reflection, and team
   contributions where applicable.
3. A **public YouTube demonstration video** linked from Devpost. The overview
   describes it as a public three-minute end-to-end demo; the Track 3 statement
   calls it a short demo and permits an API, inference, or results walkthrough
   for a backend project when a front end is not applicable.
4. Access to a functioning project, demo, or test build, free of charge and
   without restriction, through the end of judging. A public deployment is not
   itself required, but the public repository is. Any test credentials for a
   private service belong only in controlled testing instructions, never in
   this repository or its history.

The exact interpretation of “three-minute” as a target or hard maximum is not
stated. The narrower safe evidence boundary is to treat three minutes as the
maximum unless an official update says otherwise.

Sources: <https://tiktoktechjam2026.devpost.com/> and
<https://tiktoktechjam2026.devpost.com/rules>, section 4.

## Evidence implications

The rules say judges are not required to test the project and may judge only
the description, images, and video. Therefore successful benchmark execution
and self-contained submission evidence are separate requirements: runnable
access cannot substitute for clearly reported correctness, latency, setup,
limitations, and reproducibility, and presentation materials cannot establish
a benchmark run that did not occur. The Track 3 benchmark's framework,
hardware, shape, dtype, mask, tolerance, warm-up, sample count, aggregation,
and failure/skip rows are all necessary context for any defensible result.

Stage One is pass/fail viability, including the generic phrase “required
APIs/SDKs,” which Track 3 does not resolve. Stage Two has four equally weighted
criteria in this order: Technical Execution, Innovation & Problem Insight,
Feasibility & Practicality, and Impact & Relevance. That order is also the
tie-break sequence. The five-part percentage table on the Track 3 page is a
documented conflict, not the controlling formula. The Sponsor may use expert,
peer, automated AI-driven, or combined evaluation.

Source: <https://tiktoktechjam2026.devpost.com/rules>, section 6.

## Rights, provenance, and disclosure boundary

- The entry must be original entrant work, solely owned by the entrant or team,
  authorized for every third-party SDK, API, dataset, asset, and other
  component, and non-infringing. Open-source components are permitted only
  with licence compliance and an entrant-created enhancement built upon them.
- The Track 3 statement explicitly places AI-based code generation in scope
  and asks entrants to report AI skills/tools used. The rules do not create a
  separate exception for AI output: their originality, ownership,
  authorization, privacy, and third-party-assistance conditions still apply.
  Consequently, tool/version/source/licence and material-use provenance must
  be retained before any public inclusion. This is a compliance inference from
  the two official sources, not an organizer clarification.
- The two organizer benchmark attachments carry no visible licence or
  redistribution grant. A public repository is required, but that does not by
  itself grant permission to republish those files. They therefore remain only
  in this private preparation repository unless a public organizer source or
  explicit licence resolves the issue.
- Entrants retain submission IP, while submission grants the Sponsor a fully
  paid, non-exclusive judging licence. Sponsor and Devpost may publicize the
  submission and contributor name, likeness, voice, and image for three years
  after the winners announcement; submission components may be public. The
  entrant-sponsor relationship is explicitly non-confidential.

Sources: <https://tiktoktechjam2026.devpost.com/rules>, sections 4, 7, and 9;
official Track 3 statement at
<https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf#RNYvddBmXosHGbxr9jfmxYgOydd>.

## Administrative conditions recorded, not asserted

The rules require each participant to be at least 18, reside in Singapore, be
enrolled in a Singapore university with expected graduation in December 2026
or later, hold valid government identification, and satisfy sanctions and
conflict restrictions. Teams may contain at most five eligible individuals and
must appoint an authorized representative. These are requirements only; this
repository does not infer or prove any person's eligibility, registration, or
representative status. The Sponsor may request legitimate team bios after the
deadline, with a stated 48-hour response window, and prize eligibility remains
subject to identity, qualification, and authorship verification.

No Devpost draft, submission, public-repository change, credential creation,
organizer contact, registration change, or judged solution work was performed
for this audit.
