# Devpost draft — local only, not submitted

Project name: **[[FINAL PROJECT NAME REQUIRED]]**

Entrant/team attribution: **[[VERIFIED ATTRIBUTION REQUIRED]]**

Public repository: **[[PUBLIC SANITIZED REPOSITORY URL REQUIRED]]**

Public video: **[[PUBLIC YOUTUBE URL REQUIRED]]**

## One-line description

An evidence-gated Apple M5 Pro Transformer implementation that makes the
100,000-token stress case memory-bounded and accelerates all 13
reference-feasible published cases while preserving the organizer's float32
correctness contract.

## Inspiration and problem

The published Track 3 workload contains 14 causal Transformer shapes. The
largest shape uses batch 32, sequence length 100,000, model width 1,024 and 16
heads. Its explicit float32 attention-score tensor alone would require 18.626
TiB, so the supplied reference cannot run on a 64-GB personal computer. The
challenge was therefore to create a device-specific implementation that is
both fast and physically runnable, without hiding correctness failures behind
an attractive benchmark number.

## What it does

The project exports a drop-in PyTorch Transformer with the organizer's exact
parameter names and strict weight-copy boundary. It dispatches measured paths
by shape and dtype. Ordinary rows use combinations of packed Q/K/V projections,
safe internal half precision, PyTorch scaled-dot-product attention, frozen
fixed-shape graphs, and small fused Metal operations. The 100,000-token row
processes one batch item at a time and uses bounded online causal attention, so
it never constructs the quadratic score matrix.

For the stress row, a narrow Objective-C++ bridge runs MIT-licensed Apple MLX
NAX primitives directly on PyTorch MPS storage. A direct-head QKV kernel writes
all 48 Q/K/V heads without a separate layout materialization. The retained
attention specialization uses BQ256/BK48/BD64, keeps float32 online-softmax
state, and falls back safely if the Metal 4 route is unavailable.

## Measured result

On the declared 20-core-GPU Apple M5 Pro, three complete organizer-default
`high` sessions across rows 1–13 measured `2.513080x`, `2.523037x`, and
`2.508106x` versus the explicit organizer baseline, for an arithmetic mean of
**2.514741x**. All **117/117** fresh float32 comparisons passed.

The promoted direct-head QKV change for row 14 passed the S8192 organizer
reference, remained identical to the preceding champion at larger lengths,
and measured a **1.019126x** geometric-mean improvement in four balanced
S100000 sessions. The preceding safe projection route completed the full
batch-32 shape under explicit `high` in **98.588985210 seconds**, with all 3.2768
billion outputs finite.

**Promoted-route contention-controlled result:** the three exact batch-32 runs completed in **54.269, 51.896, 52.003 seconds**, with a **52.003-second median** and a 51.896–54.269-second range. All returned values were finite, no external high-CPU Python, Node or Codex process was observed by the one-second monitor, and power state was not inspected or used as a gate. The 98.588985210-second point remains separately labelled as the preceding route. No official MFU or combined organizer score is claimed.

## How it was built

The work followed a correctness-before-timing loop: inspect a source-backed
idea, build an isolated candidate, compare against the organizer predicate,
measure baseline and candidate in alternating same-session order, promote only
repeatable winners, and preserve every rejection. This led from PyTorch/MPS
profiling to bounded Metal attention, then to Apple's Metal 4 NAX tensor
operations for the dominant long-sequence path.

The retained third-party sources are pinned to exact commits and preserve their
MIT licences. Six production inputs are SHA-256 locked. A deterministic
public-safe release excludes the organizer's unlicensed attachments, private
Git history, caches and raw outputs. It has been installed without a package
cache, compiled from empty native caches, and exercised through its packaged
test suite.

## Why it matters and why it is practical

The largest published workload is not merely slow under the explicit
formulation: its attention matrix cannot fit on the declared 64-GB personal
computer. The bounded route changes that exact challenge shape from physically
unrunnable into a complete 100,000-token execution, with a 52.003-second local
median for the promoted route. This demonstrates a practical path for running
long-context Transformer experiments on one Apple Silicon development machine
instead of requiring a CUDA datacenter system for this published suite.

The result remains usable as a PyTorch module rather than a disconnected
microbenchmark. It preserves the organizer's parameter interface, has explicit
fallbacks, documents approximately 31 GiB of unified-memory driver allocation
for the stress row, and was rebuilt from an uncached environment and empty
native cache. Its demonstrated impact is deliberately limited to the published
workload and declared M5 Pro; broader model or production-inference impact is a
potential application, not a measured claim.

## Challenges

- The explicit row-14 reference is physically infeasible, requiring scaled
  explicit checks and full-length continuity evidence instead of pretending to
  have full-reference equality.
- Apple GPU timings vary strongly across sessions, so promotions use paired
  same-session comparisons. The final absolute protocol records memory and
  swap, polls for external high-CPU Python, Node and Codex contention once per
  second, and
  requires a 60-second clean window before every run, rejects any monitored
  contender during the measured interval, and applies a three-run max/min ratio
  no greater than 1.15. It does not inspect or gate on power state.
- Several plausible ideas failed: complete MLX execution narrowly missed the
  tolerance, approximate sparsity failed correctness, unsafe tiles were
  repaired or rejected, and many faster-looking microkernels lost after full
  integration.
- The organizer attachments have no visible redistribution licence, requiring
  a new allowlisted public history rather than exposing the private repository.

## Accomplishments

- Every one of the 13 reference-feasible rows is faster in the complete score
  sessions and passes the organizer predicate.
- The 100,000-token row is bounded and completes without quadratic attention
  memory.
- The fastest retained stress path materially uses Apple-native NAX kernels
  while remaining a PyTorch-compatible drop-in with safe fallbacks.
- Performance, correctness, source provenance, AI assistance, timing variance,
  failed experiments, and publication boundaries are independently
  documented rather than compressed into an invented score.

## What was learned

Open-source projects were most valuable as books, not wholesale solutions. A
full MLX rewrite was not accurate enough, but Apple's lower-level NAX template
became the decisive building block when adapted directly to PyTorch MPS
buffers. Small component wins also did not reliably transfer to the complete
Transformer, making full-model paired measurement more important than isolated
kernel timing.

## What's next before submission

1. Choose the final project name and verify entrant/team attribution rather
   than inferring either from private context.
2. The prior local video draft is rejected and is not a submission candidate.
   Video generation is paused. When explicitly resumed, create fresh
   ElevenLabs narration with a newly rotated runtime-only credential, keep the
   credential outside files and logs, and require a complete human watch before
   treating any replacement as usable.
3. Replace every bracketed action-time field above with verified facts and
   public URLs.
4. Rebuild the already-proven one-commit public history from the final verified
   sanitized tree after all approved content changes.
5. Record the final AI/tool accounting and exact host-visible model label if
   one is available.
6. Publish and submit only after separate action-time authorization, then
   verify the public URLs and Devpost receipt.

## Built with

Codex AI coding agent (exact host-visible model label unavailable), Python 3.9,
PyTorch 2.8, Apple MPS, Metal 4, Apple MLX NAX, Objective-C++, and Ninja.
