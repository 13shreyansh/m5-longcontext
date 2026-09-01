# Correctness-oracle audit

Snapshot: **2026-08-26 SGT**

Public-release note: the 2026-08-27 PyTorch attachment changed only its CLI
defaults to `rtol=0.02` and `atol=0.002`. The comparison function itself is
unchanged. The probe below deliberately retains explicit `0.01` / `0.001`
arguments as evidence of the archived 2026-08-26 snapshot; it is not a claim
about the current CLI defaults.

This preparation-only audit describes and probes the comparison routines in
the two unmodified organizer attachments. It does not modify a benchmark or
implement a judged Transformer.

## Exact predicates

For every output element, both scripts require finite reference and candidate
values plus an **OR** condition:

```text
absolute branch OR relative branch
```

Archived 2026-08-26 PyTorch defaults used by this probe:

```text
abs(candidate - reference) <= 0.001
OR
abs(candidate - reference) <= 0.01 * abs(reference)
```

TensorFlow defaults:

```text
abs(candidate - reference) < 0.002
OR
abs(candidate - reference) <= 0.02 * abs(reference)
```

Thus the effective permitted error is the larger branch threshold, with the
TensorFlow absolute branch remaining strict. Both frameworks cross from
absolute-dominated to relative-dominated tolerance at `abs(reference)=0.1`.
Every element must pass; matching NaN or infinity still fails because values
must be finite.

The reported `max_abs_error` and `max_relative_error` are independent
diagnostics over all elements, not conjunctive acceptance tests. A passing
vector can therefore report maximum absolute error above `atol` and maximum
relative error above `rtol`: different elements may pass different OR
branches. Headline maxima must be read together with failed-element count and
the exact predicate.

## Reproducible boundary probe

Command:

```bash
.venv/bin/python scripts/probe_correctness_oracles.py
```

Observed exit code: `0`.

Observed result lines (the already-recorded unrelated urllib3/LibreSSL warning
is omitted):

```text
torch:or_branches: passed=True failed=0/2 max_abs=0.0089999437 max_rel=0.018000007
torch:zero_at_atol: passed=True failed=0/1 max_abs=0.001 max_rel=1.0000001e+09
torch:matching_infinity: passed=False failed=1/1 max_abs=nan max_rel=nan
tensorflow:or_branches: passed=True failed=0/2 max_abs=0.019000053 max_rel=0.037999973
tensorflow:zero_at_atol: passed=False failed=1/1 max_abs=0.0020000001 max_rel=2.0000001e+09
tensorflow:matching_infinity: passed=False failed=1/1 max_abs=inf max_rel=inf
```

The matching-infinity TensorFlow case also emitted `RuntimeWarning: invalid
value encountered in subtract` from the unmodified comparison function before
returning the recorded failure. The process still exited `0` because the probe
itself intentionally reports oracle results rather than treating an expected
synthetic failure as a script error.

The zero-reference relative diagnostics are large because both scripts divide
by `max(abs(reference), 1e-12)` only for reporting. That denominator is not
used by the pass predicate. The boundary difference is confirmed: PyTorch
accepts exact `atol`, while TensorFlow's absolute branch rejects exact `atol`
when the relative branch is zero.

## What the default accuracy trials cover

PyTorch runs five default trials for its single selected shape. Inputs are
normal random tensors scaled by `input_scale`, with trial seeds `1234` through
`1238` under defaults. TensorFlow runs three trials per generated case, with
case-dependent stateless seeds. Neither default enables causal attention or
requested padding; both generators still pass an all-true mask, so the
`valid_token_mask=None` path is not tested.

Padding tests, when explicitly requested, use only contiguous valid prefixes,
never a fully invalid sequence, and yield approximately half the requested
padding ratio on average because valid length is sampled uniformly between the
minimum and full sequence length. Timing uses a different fixed random input
and does not run the accuracy oracle on that timed input.

The oracles perform one baseline call and one candidate call per sampled input.
They do not test gradients, training, repeated-call determinism, arbitrary
mask topology, adversarial values, or semantic equivalence outside sampled
points. Passing the supplied trials is evidence for those samples only, not a
proof of the full formula or undisclosed tests.
