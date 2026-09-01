# Upstream experiment audit — 2026-08-29 SGT

This audit was performed after the official challenge window opened. All
third-party checkouts, build trees, package caches, virtual environments, and
outputs are under ignored `artifacts/`; no model weights were downloaded and no
credentials were stored. “Pass” below means the named command exited `0`.

## Live organizer re-check

At 12:30–12:31 SGT, the authenticated Lark document displayed `Last updated:
Aug 28`. Track 3 still displayed its explicit `27 August 2026, 6:25PM` update
notice. The current downloads were byte-identical to the preserved files:

| Attachment | Bytes | SHA-256 | `cmp` |
|---|---:|---|---:|
| `torch_transformer_benchmark.py` | 25,017 | `5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e` | 0 |
| `tensorflow_transformer_benchmark.py` | 54,531 | `00e99b6e1d19e961039b66eb3d3c055b36cc50f0436da2558f5f1fbe292ef798` | 0 |

The visible August 28 addition is the Track 3 webinar recording:
<https://bytedance.larkoffice.com/wiki/TesdwVQkDiIJCokHobTcdM5jnch>.

## Source and licence inventory

Every checkout was shallow-cloned from the listed URL. All materially reused
sources are MIT and retain their copyright and licence notices. One audit-only
checkout is Apache-2.0; it was not reused. One newly screened source has no
licence file and was not reused. The organizer's
originality/enhancement rules remain an independent boundary.

| Source | Commit/tag | Licence SHA-256 |
|---|---|---|
| [Apple MLX](https://github.com/ml-explore/mlx) | `3f0bd54ff0c0af5b88530191d5df31010ce54fcd` | `ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01` |
| [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) | `8671cddc38f19a6eadb804dee6a3ca2954b8bf32` | `84c1ba93965bfb1e2e900a082bbe6beae3c324395d091ca83bf2e9ecab1c2e9c` |
| [triton-msl](https://github.com/bledden/triton-msl) | `182c1820fd24a836d565e1da842f28414de64084` / `0.2.0` | `4b35b714917b654cd6a921723a747be7b02178759a4f42e9a322d185a71a969d` |
| [mps-flash-attention](https://github.com/mpsops/mps-flash-attention) | `39c2ba51cd009d02c0aa8c9b46ac7db2d1385e77` / `v0.6.1` | `17f5e65d269bd8efa11dca8ba582567856aa07a180fea89310ddb7a7655f6565` |
| [mlx-flashattention-steel](https://github.com/marcogva-hub/mlx-flashattention-steel) | `b1e7ac2586fde9e28412d81b62540f0bff281771` / `v2.62.1` | `e60a41d3aada1d8ec9156d4a2ea564a3bb57d850c1665a88b6db105387c22a28` |
| [mlx.fast Gemma engine](https://github.com/Layr-Labs/mlxfast-gemma4-26b-a4b-engine) | `bdbb9947227e52e4ae2664927695da1c11129050` | `3e4515c8baee16121ff56e5d6cdf9e0b70b3db31cacdf1983ab4d813b5427f04` |
| [mlx-train-perf](https://github.com/IonDen/mlx-train-perf) | `cdfce970372bd3d78aab1d6f09d25e7f5f65a509` / `v0.6.0` | `9cd05bda9eb2f468c2ce50c6cb93a037a1dc143e7e2f94ef0202fb8917c60d89` |
| [mlx-metal-kernels](https://github.com/manishklach/mlx-metal-kernels) | `9fc1d38a0076a017ac8521653d3ab469abcc22bf` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` (Apache-2.0; no reuse) |
| [ComfyUI-SolAttn-MPS](https://github.com/yshenaw/ComfyUI-SolAttn-MPS) | `45071126b0c1ee30b0e6b7103fa9d70924828ba5` | `83ff744062652b24cd1e223599786d2ce81c7eda6765e05fa940aa4946eb6646` |
| [flash_attn_metal_cpp](https://github.com/harvestingmoon/flash_attn_metal_cpp) | `f91c9b182549c3fa8c4076c80182231295fbf363` | **No licence file found; no reuse** |

The last repository is the engine linked by the referenced
[tweet](https://x.com/rachpradhan/status/2092472018352644317). It is a separate
hill-climbing methodology source, not Apple MLX's `mx.fast` namespace and not
one of the five direct implementation candidates.

## Isolated runtimes

The repository-local installer/runtime commands were:

```bash
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/artifacts/tools/uv-bin" sh
UV_CACHE_DIR="$PWD/artifacts/uv-cache" UV_PYTHON_INSTALL_DIR="$PWD/artifacts/python-install" artifacts/tools/uv-bin/uv python install 3.14 3.12
```

Observed: uv `0.12.7`, CPython `3.14.7`, and CPython `3.12.14`. The uv installer
created two convenience links in `~/.local/bin`; both exact links were removed
immediately, leaving the runtimes inside ignored repository storage. The test
environments used PyTorch `2.8.0` and `2.13.0`, Triton `3.7.0`, MLX `0.32.0`,
and `mlx-mfa` `2.62.1`. Total ignored artifact use after builds was 6.8 GiB;
717 GiB filesystem space remained.

## Candidate results

### Apple MLX

MLX `0.32.0` ran its GPU SDPA successfully. Representative float32 calls took
approximately 0.24–0.62 ms in the small standalone probe, but these figures are
not directly comparable with the official PyTorch model. PyTorch `2.8.0`
returned `RuntimeError: Cannot pack tensors on mps:0` from
`torch.utils.dlpack.to_dlpack`, proving there is no zero-copy PyTorch-MPS to MLX
bridge in this environment. A framework copy would dominate small cases and
inflate memory use, so the complete MLX runtime was rejected as the direct
submission path.

The lower-level NAX attention source was later tested independently. The exact
half/BQ64/BK32/BD64/WM4/WN1 template compiled in Metal language 4.0 and ran
directly on PyTorch MPS buffers through a small Objective-C++ bridge. It passed
an explicit 8,192-token whole-model reference check, four 100,000-token
continuity seeds, a fresh 39/39 ordinary matrix, and the exact batch-32 run.
Four production A/B sessions produced a 3.164973x geometric-mean gain over the
preceding champion. Eight exact MLX headers and Apple's MIT notice are retained
and byte-verified; details are in `SOLUTION_MILESTONE_17_2026-08-29.md`.

### Philip Turner metal-flash-attention

```bash
swift build -Xswiftc -Ounchecked
swift test -Xswiftc -Ounchecked
```

The library build passed. The test command failed because the Command Line
Tools Swift distribution has no `XCTest` module. A later standalone executable
removed that test-framework dependency and invoked the public forward-kernel
API directly. The generated Metal source then failed before pipeline creation:
macOS 26 rejects the private `air.simdgroup_async_copy_*` and
`air.wait_simdgroup_events` assembly declarations under both the default
compiler mode and an explicit Metal 3.1 language version. The explicit command
exited `133`. Removing only the declaration header was not valid because nine
`simdgroup_event` uses remained in the body. The released API is also
single-head/non-causal and has no PyTorch adapter. Consequently no runtime or
performance claim exists; the exact probe and resource use are in
`SOLUTION_MILESTONE_118_2026-08-30.md`.

### mps-flash-attention

The Swift bridge and editable PyTorch package built successfully against Python
3.12/PyTorch 2.8, and `is_available()` returned true. Every isolated launch
tested (`N=128,D=32/64`, float16/float32, causal/non-causal, and
`N=1024,D=64`) terminated with exit `133` at
`AttentionDescriptor+PipelineCache.swift:121`, reporting
`XPC_ERROR_CONNECTION_INTERRUPTED` from the Apple Metal compiler service.
Therefore no correctness or performance result exists for this host.

### Current public-source refresh

`ComfyUI-SolAttn-MPS` at `45071126...` compiled its unmodified Metal source in
the already isolated PyTorch `2.15.0.dev20260829` runtime. The compile command
exited zero in 4.89 seconds and used 230,359,040 bytes maximum RSS. The source
implements approximate sparse block routing, explicitly falls back for masked
attention, and contains no causal branch. Its BQ32/BQ64, BK64, shared-staging
design also overlaps local designs already rejected against M5 NAX. It is a
useful MIT source, but not a valid implementation of the published dense
causal row; no attention output was timed or promoted.

`flash_attn_metal_cpp` at `f91c9b1...` contains no licence file. Its optimized
kernel documents causal failures, while its verifier allows a 0.06 causal
maximum-error threshold and its safe kernel uses scalar/`float4` work rather
than M5 tensor operations. It was not built, run or reused. Commit, tree,
archive checksums, compile evidence and the source-grounded closure are in
`SOLUTION_MILESTONE_120_2026-08-30.md`.

`mlx-train-perf` at `cdfce970...` is MIT-licensed training-memory
infrastructure. Its only measured saturated forward tile is
`B1/Hq32/Hkv8/N8192/D128/bf16`; its D64 classic-SIMDgroup route is explicitly
provisional, and its reported full-training comparisons trade 5.3–5.9%
throughput for lower memory. The Track-3 row is B32/H16/N100000/D64 inference
and already uses an M5 NAX forward, so the source was closed without install,
build, model allocation or timing. Exact hashes and reasoning are in
`SOLUTION_MILESTONE_135_2026-08-30.md`.

`mlx-metal-kernels` at `9fc1d38a...` is an Apache-2.0 correctness-first
laboratory whose own checkout publishes no Apple benchmark numbers. Its
default path is one thread per row; the experimental D64 alternatives still
iterate keys sequentially with scalar, threadgroup or SIMD reductions and use
no `simdgroup_matrix` or Metal-4 NAX operations. Its fp16 benchmark accepts
`atol=0.02`, ten times the organizer absolute tolerance. It was closed without
install, build, tensor allocation or timing. Exact source hashes and the
decision boundary are in `SOLUTION_MILESTONE_141_2026-08-30.md`.

### triton-msl

Python 3.14, PyTorch 2.13, the supplied Triton 3.7 wheel, and `triton-msl`
0.2.0 installed successfully. The full Triton FlashAttention test file produced
41 failures because no Triton Metal driver activated; `xcrun --find metal`
fails on this host. Its standalone zero-copy templates do not require that
driver:

```bash
python -m pytest -q tests/test_fa_simdgroup_template.py tests/test_fa_tiled_template.py
```

Observed: `18 passed in 6.16s`. Direct official-tolerance probes passed for
head dimensions 64 and 128, while the simd template for head dimension 32 did
not compile (`zero-length arrays are not permitted in C++`). A five-round,
alternating-order probe on exact published attention shapes found the custom
template slower than PyTorch SDPA: 0.78x at `(B=64,H=1,S=128,Dh=128)`, 0.71x at
`(64,2,128,64)`, and 0.53x at `(32,16,128,64)`. These local measurements do not
reproduce the README's M4 performance claims and are the applicable result for
this M5 Pro/PyTorch 2.13 stack.

The documented `torch.compile(..., backend="metal")` route did not register on
PyTorch 2.13 (`InvalidBackend`), so it is not ready for the official model here.

### mlx-flashattention-steel

The first build against current MLX `0.32.2` correctly refused an unknown
nanobind ABI. Pinning MLX `0.32.0`, which the project explicitly maps, and
building with `--no-build-isolation` passed. The full suite reported:

```text
2 failed, 3525 passed, 113 skipped, 1 xpassed in 461.31s
```

Both failures were missing-optional-dependency errors for `mlx_lm`. After
installing `mlx-lm 0.31.3`, both exact failed tests passed. No numerical test
failed. On this M5-class host, float32 `backend="auto"` deliberately falls back
to Apple MLX SDPA and forced `backend="mfa"` refuses float32. In sampled fp16
tests the native path was slower than MLX SDPA (0.330 vs 0.206 ms at
`B1,H4,S128,Dh128`; 1.848 vs 0.685 ms at `S2048`). Combined with the framework
copy boundary, this is an excellent routing/design reference but not a direct
official-PyTorch candidate.

### mlx.fast engine methodology

`swift build` passed. `swift test` compiled the engine and vendored MLX sources
but failed at the test target because the local Swift distribution has no
`Testing` module. No 15.6 GB checkpoint or 40 GB working set was downloaded.
The reusable ideas are paired same-session baselines, correctness-before-
timing, thermal gates, repeated pairs with robust aggregation, explicit
editable surfaces, and promoting only a verified winner. The checked-out
README and machine-readable fixture disagree about whether DFlash is enabled;
the fixture and participant contract say it is enabled, demonstrating why
machine-readable gates must outrank prose.

## Whole-Transformer SDPA experiment

[`../experiments/benchmark_sdpa_candidate.py`](../experiments/benchmark_sdpa_candidate.py)
imports the unmodified organizer module, swaps only explicit attention for
PyTorch SDPA, uses the organizer's exact elementwise OR predicate, alternates
measurement order, and synchronizes MPS around host timing. Example command:

```bash
.venv/bin/python experiments/benchmark_sdpa_candidate.py --case 1 --device mps --warmup 2 --repeats 5 --rounds 3
```

All float32 published-shape cases 1–13 passed with `0` differing output
elements. Measurements are local M5 Pro direction-finding results, not
organizer scores:

| Case | `(B,S,D,H,F,L)` | Baseline ms | SDPA ms | Speedup |
|---:|---|---:|---:|---:|
| 1 | `(64,128,128,4,128,4)` | 9.619 | 6.574 | 1.463x |
| 2 | `(1,128,128,4,128,4)` | 1.795 | 1.715 | 1.047x |
| 3 | `(4,128,128,4,128,4)` | 1.580 | 1.472 | 1.074x |
| 4 | `(16,128,128,4,128,4)` | 3.290 | 2.596 | 1.267x |
| 5 | `(128,128,128,4,128,4)` | 17.371 | 11.868 | 1.464x |
| 6 | `(10000,128,128,4,128,4)` | 1887.201 | 1299.875 | 1.452x |
| 7 | `(64,128,32,4,32,4)` | 6.100 | 3.338 | 1.827x |
| 8 | `(64,128,1024,4,1024,4)` | 143.274 | 131.771 | 1.087x |
| 9 | `(64,128,128,1,128,4)` | 5.367 | 4.899 | 1.096x |
| 10 | `(64,128,128,2,128,4)` | 7.148 | 5.772 | 1.238x |
| 11 | `(64,128,128,16,128,4)` | 24.076 | 11.403 | 2.111x |
| 12 | `(64,32,128,4,128,4)` | 1.768 | 1.706 | 1.036x |
| 13 | `(64,1024,128,4,128,4)` | 434.482 | 254.694 | 1.706x |

Cases 2, 3, and 12 were re-measured with 10 warm-ups, 30 repeats, and seven
alternating rounds after short runs produced unstable results. Their revised
1.036–1.074x margins remain small and should be revalidated before dispatch.
On case 2 with 25% padding, the exact organizer predicate passed for float32
(`0` failed elements, 1.095x) and float16 (`0` failed elements, 1.236x), but
bfloat16 failed (`576/16,384` elements, max absolute error `0.046875`). A
second bfloat16 correctness run without padding also failed (`711/16,384`).
SDPA is therefore not a valid bfloat16 replacement under the current evidence.
Case 6 used one measured sample per leg after its exact correctness pass, so
its timing confidence is lower than the other rows. Case 14 was not allocated:
the explicit reference score tensor is 19,073.486 GiB (18.626 TiB), far above
the experiment's 16 GiB safety cap and this machine's 64 GB memory. The command
exited `3` with a `SKIP` record. This is a concrete reproduction blocker, not a
failed optimization.

## Conclusion from the evidence

The best immediately usable primitive is PyTorch SDPA behind a conservative
shape-and-dtype dispatcher, with the explicit baseline retained for bfloat16
and a repeat-measurement gate for small-margin cells. The external Metal
projects contribute tile, online-softmax, and fail-closed
routing patterns, but none beat PyTorch SDPA on the exact sampled official
shapes in a compatible zero-copy path. The larger opportunity is the rest of
the block—QKV projection, LayerNorm/residual, GELU/FFN—and a memory-bounded
strategy for the 100,000-token row. Any custom kernel must earn its place
through the same paired, synchronized correctness-and-timing gate.
