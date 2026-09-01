# Track 3 optimized Transformer

`optimized_transformer.py` exports `UserOptimizedTransformer`, a drop-in class
with parameter names identical to the organizer PyTorch baseline. Its current
dispatcher uses SDPA/packed QKV for ordinary shapes, a fused residual-plus-
LayerNorm route for the width-1024 float32 case, and bounded case-14 paths that
never materialize the full attention-score tensor.

For float32 sequences of at least 32,768 tokens, the packed Q/K/V projection
uses a lazy fp16 weight/bias cache. A fail-closed Metal 4 bridge instantiates
Apple MLX's NAX linear primitive as BM32/BN512/BK256/WM1/WN8 and stores its
output directly as the 48 contiguous head-major Q/K/V tensors needed by
attention. Four independent S100000 sessions passed bit-identically and
measured 1.011633x/1.021096x/1.021071x/1.022742x paired means over the prior
packed-half projection plus layout route. Unsupported inputs or a native build
or launch failure fall back to that prior route.

The attention result returns to fp32; output projection, residuals,
normalization, FFN, input, and output remain float32. The half Q/K/V feeds a
pinned Apple MLX NAX register-tiled attention specialization through a small
PyTorch MPS custom bridge. The retained instantiation uses a 256-query by
48-key tile, sixteen SIMD groups, Metal 4 tensor operations, fp32 accumulation,
and fp32 online-softmax state. The target route pads Q by 96 rows to make the
100,000-token input a complete BQ256 launch, keeps K/V at their original
length, and discards the padded outputs; this enables the aligned-Q
specialization without changing any retained output. The kernel keeps
the four BD64 query fragments resident, applies the log-base-adjusted scale once
to Q, and removes the per-score scaling loop. Five complete-model sessions
were bit-identical to Q-prescaled BQ128 and measured a 1.072758x geometric-mean
speedup. The BQ128 predecessor's Q-prescale promotion measured 1.030260x over
aligned-Q. That aligned predecessor's four
aligned/unaligned sessions measured 1.022838x. Four earlier BQ64/BQ128 sessions
measured a 1.029597x incremental mean. Two fresh direct aligned-Q sessions
versus the preceding exact-constant stress champion measured a 3.270180x
geometric mean. If Metal 4,
the local compiler, or the bridge is unavailable, the byte-verified
triton-msl-derived champion remains the automatic fallback.

The first integrated BQ64 NAX route completed the exact published batch-32,
100,000-token float32 shape in 81.266 seconds with every one of 3,276,800,000
outputs finite and 30.489 GiB MPS driver allocation. A separate BQ64 repeat
also passed every output but took 160.192 seconds. The unaligned BQ128 route
completed an exact run in 115.640 seconds, the aligned-Q predecessor completed
in 146.616 seconds, and Q-prescaled BQ128 completed in 116.184 seconds. These
cross-session historical observations are retained as completion/resource
evidence, not pooled as causal speed evidence or labelled official MFU.
The preceding packed-projection safe BQ256/BK48 route completed in 69.605,
97.173 and 131.088 seconds, with an organizer-default `high` point at 98.589
seconds, the same 30.489 GiB driver allocation and all outputs finite. The new
direct-head QKV route completed the final contention-controlled exact B32
protocol in 54.269, 51.896 and 52.003 seconds, with a 52.003-second median,
1.045733 max/min ratio, every output finite and zero monitored contenders.
Each run followed a 60-second clean process window and was monitored once per
second; power state was not inspected or used as a gate. Earlier incomplete or
contended 56.248–60.890-second observations remain excluded rather than pooled
into this set. Paired same-session ratios remain the attribution boundary, and
the preceding route's 98.589-second point remains separately labelled.

## Verification

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_optimized_transformer.py
.venv/bin/python scripts/verify_solution_provenance.py
.venv/bin/python experiments/benchmark_sdpa_candidate.py \
  --candidate solution --case 1 --device mps --dtype float32 \
  --accuracy-trials 5 --warmup 8 --repeats 15 --rounds 4
.venv/bin/python experiments/run_case14_solution.py \
  --batch-size 1 --seq-len 100000 --dtype float32 --seed 1401
```

The first command verifies strict state-dict compatibility, packed-weight
refresh, all-valid and padded mask routes, all three declared dtypes, both
base Metal-source compilation paths, the derived fast-exp source and exact
bfloat slicing, a real small NAX launch, the exact 32,768-token dispatch
boundary, unsupported-input rejection, non-default batch/head support, and
both NAX-unavailable and non-divisible-length fallbacks. The second regenerates the two base
shaders, byte-matches all eight vendored MLX headers against pinned upstreams,
verifies asserted transforms, and verifies both retained licences. The third
performs the organizer-shaped correctness and synchronized timing gate. The
fourth runs the solution at the target sequence length without constructing the
infeasible organizer reference.

The generated float32/float16 attention sources and fused normalization source
are adapted/generated from triton-msl commit
`182c1820fd24a836d565e1da842f28414de64084`. The retained MIT licence is at
`third_party/triton-msl-LICENSE`; the generated modules record their exact
specializations.

The NAX source is copied from Apple MLX commit
`3f0bd54ff0c0af5b88530191d5df31010ce54fcd`. The retained Apple MIT licence is
at `third_party/mlx-LICENSE`. Building the tiny Objective-C++ bridge requires
the pinned `ninja==1.13.0` dependency, Command Line Tools, macOS 26, and an NAX-
capable Apple GPU; the dispatcher falls back cleanly when those conditions are
not met.

An experiment-only `TRACK3_NAX_TELEMETRY` compile macro attaches public Metal
completion timestamps at each NAX dispatch. Production never defines this
macro; telemetry measurements are diagnostic and are not score evidence.

The solution is inference-only. Packed buffers are non-persistent and are
refreshed whenever `load_state_dict` is called. Milestones 17–37 retain the
NAX dispatch-boundary audits and the BQ128, aligned-Q, Q-prescale and BQ256
promotion history. The current direct-head QKV champion is recorded in
`docs/SOLUTION_MILESTONE_149_2026-08-30.md`; the current contention-controlled
measurement policy and completed three-run result are recorded in
`docs/SOLUTION_MILESTONE_207_2026-08-31.md` and the champion manifest.
No submission action is performed by these files.
