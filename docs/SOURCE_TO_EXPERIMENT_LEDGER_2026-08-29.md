# Source-to-experiment ledger — 2026-08-29 SGT

This ledger separates reading a useful open-source project from earning a place
in the Track 3 result. MIT permission permits reuse with its notice; it does not
make an upstream result valid on this machine, prove organizer correctness, or
satisfy the organizer's request to implement and optimize rather than submit an
existing project unchanged. A source-derived idea is promoted only after an
exact-predicate correctness check and synchronized same-machine timing.

## Organizer facts that control the strategy

The August 28 webinar transcript says that participants should optimize for
their own machine, explicitly gives an Apple unified-memory Mac as an example,
and says judges will not rerun the benchmark because implementations are
device-specific. Every published shape must first pass float32 input/output
precision; internal quantization is allowed. The technical result is described
as a weighted combination of per-case MFU, with memory bandwidth considered.
Compilation and the first run may be excluded. The exact weights and MFU formula
remain unpublished, so local speedup is evidence, not a claimed official score.
The selected 20-core-GPU M5 Pro has a vendor-published 307 GB/s unified-memory
bandwidth; Apple does not publish a conventional fp32 peak-FLOP denominator in
the product specification, so the report will provide the bandwidth and
derived operation rate without fabricating an MFU percentage.

Consequently, the Apple M5 Pro is the target machine. Absence of CUDA is not a
competition blocker. PyTorch compatibility remains useful because it gives the
cleanest comparison with the organizer script, but a reproducible MLX- or
Swift-native alternate is also eligible if it uses identical inputs and weights,
produces a precision-passing output, and records an honest timing boundary.

## What each source has contributed

| Source (“book”) | What was actually read/tested | What entered the current solution | What did not earn promotion | Next evidence-gated use |
|---|---|---|---|---|
| Apple MLX, `3f0bd54...` | Built/running MLX `0.32.0`; tested GPU SDPA, DLPack, and a full MLX-native Transformer; then inspected and compiled its NAX attention template with Metal 4 and launched it directly on PyTorch MPS buffers through a small Objective-C++ bridge. A source-supported tile sweep advanced BQ128 over Apple's BQ64 default; one-time Q prescaling then enabled BQ256/BK32/BD64/WM16/WN1, and a later gap screen promoted BK48 for the exact stress route. A source audit then found BK48's odd three-fragment score tile had an invalid fourth paired-operation destination; five safe forms were compiled and gated. Later current-source routing also motivated a direct float32/TF32 instantiation of the same safe structure. | **Material reuse:** eight pinned Apple MLX NAX headers, the half attention template, Metal-4 language selection, a locally verified 512-thread BQ256/BK48 instantiation, aligned-Q padding, resident Q fragments with one-time prescaling, partial-key handling, bounds-safe rounded BK48 score storage with three-column reductions, and automatic toolchain fallback. Apple MIT notice and byte-level provenance checks are committed. The BK48 promotion has a 1.082579x five-session complete-model mean over BK32; the safety repair is bit-identical with a neutral 0.998777x end-to-end paired mean. | PyTorch→MLX DLPack still has no zero-copy path, and the complete MLX runtime still narrowly fails the organizer predicate; neither route is used. `torch.mps.compile_shader` also cannot compile NAX because it selects an older Metal language, so the narrow bridge is required. BQ32, BK64, BQ128/BK64, head split, shared K/V staging, reduced unrolling and alternate P@V order lost their gates. Direct token-major output was neutral; TQ2/BK64 0.389657x; true TQ2 K/V-fragment reuse 0.937859x versus production; float32/TF32 NAX 0.574586x; inner-barrier removal 0.980449x; half probabilities 0.937760x; NAX strided Q/K/V 0.395871x; reverse scheduling reversed from 1.022960x isolated to two negative complete sessions; half per-score exponential timing reversed between bounded sessions; the zero-copy BK32-prefix/BK48-tail split fell to 0.997634x/1.001641x under balanced target ordering; and current-state Q@K/P@V interchanges measured 0.996563x/0.997218x (0.971830x together). | Retain only the verified causal head-64, sequence-divisible-by-32 stress envelope and keep the preceding triton-msl-derived champion as fail-closed fallback. |
| Philip Turner `metal-flash-attention`, `8671cdd...` | Read the tiled attention/GEMM generator, parameter tables and SIMD-shuffle softmax reductions; built the Swift package; then wrote a standalone public-API executable to remove `XCTest` from the runtime path. | Online-softmax, explicit tile specialization, precision control and device-specific dispatch informed the bounded long-sequence design. | The released source is single-head/non-causal and has no PyTorch adapter. Its generated Metal depends on private AIR SIMD-group async-copy entry points that macOS 26 rejects under both default and explicit Metal 3.1 compilation; the probe exits 133 before pipeline creation. Removing the header is invalid because nine event uses remain. A 16-query-row derived variant also failed correctness; a later 8-lane SIMD-local reduction passed correctness but changed timing sign. | Close direct use. Reconsider only if upstream replaces the private AIR primitive and provides causal support; never infer performance from its older-device tables. |
| `triton-msl`, `182c182...` | Tested the standalone generators, generated kernels, compiled them through `torch.mps.compile_shader`, and ran paired accuracy/timing experiments, including the source-supported `half_accumulate=True` path, an asserted four-site Metal fast-exponential transform, and exact runtime-constant binding. | **Material reuse:** the current float32/float16 case-14 online-attention shaders, the fast-exp stress variant, compile-time stress constants, and the fused residual-plus-LayerNorm kernel are derived/generated from this source. Exact MIT notice, commit, regeneration, source-transform, and ABI-preservation checks are committed. | Its small-shape attention template lost to PyTorch SDPA; its full Triton Metal backend did not activate; fp16 MMA accumulation failed 12,066,514 target outputs; and a larger-tile rearrangement failed correctness. | Retain fp32 accumulation/state, the fast-exp variant, and constant binding keyed by exact runtime shape/strides; test only one source-grounded attention-core change at a time. |
| `mps-flash-attention`, `39c2ba5...` | Built its bridge/package and attempted multiple causal/noncausal launches. | Failure-mode knowledge and fail-closed routing. | Every tested launch crashed in the Metal compiler-service path, so it has no correctness or performance result and contributes no submitted kernel. | Revisit only if the compiler-service blocker is resolved; do not spend core challenge time retrying unchanged launches. |
| `mlx-flashattention-steel`, `b1e7ac2...` | Ran its large test suite and resolved the optional `mlx_lm` dependency; inspected routing and M5 release gates; benchmarked native attention against MLX SDPA; then read its NAX FFN and SageAttention sources. Screened all eight published linear tile families on exact M100000/N1024/K1024 through MLX and direct PyTorch MPS, generalized the bridge to packed-QKV N3072, and tested both a direct current-NAX block-int8 K adaptation and the unmodified cooperative Sage kernel at causal D64. | Shape/dtype dispatch, machine-readable gates, explicit fallbacks and measurement discipline informed our workflow. The experiment-only NAX-linear and int8-K harnesses retain the upstream MIT provenance; no upstream dense or Sage code entered production. | Forced native attention refuses float32 and sampled fp16 lost to MLX SDPA. Padded native NAX linear beat MLX by 1.223567x, but the relevant direct-PyTorch N1024 sweep was bit-identical and every tile lost at 0.882660x–0.968878x; the unpadded M100000 upstream path exposed an invalid tail. N3072 reached only 1.062929x, about 0.22% optimistic whole-row value. Sage-style K quantization passed five S8192 organizer references, but the bit-identical direct int8 loader measured 0.778038x before and 0.681579x after quantization; upstream cooperative Sage reached only 0.19x/0.23x versus fp16 at S8192/S32768. | Retain the source-backed rejections. Revisit linear only with a bridge-free relevant kernel above the 1% whole-row threshold, and int8 K only with a materially different Apple tensor/conversion primitive. |
| `mlx.fast` Gemma engine, `bdbb994...` | Read its participant contract, measured-window definition, scoring aggregation, thermal/cool gate, editable-surface rules, and correctness gates; `swift build` passed. | Paired same-session baseline/candidate measurements, correctness-before-timing, thermal awareness, robust aggregation, explicit editable surfaces, and winner-only promotion are already reflected in the experiment workflow. | The engine itself targets a different Gemma decoding benchmark and a 15.6-GB checkpoint; importing it would not implement this Transformer task. Its tests also require an unavailable Swift `Testing` module. | Turn its methodology into a Track-3 experiment registry: exact candidate, shape, seed, temperature/order, correctness, median/dispersion, decision, commit, and rollback record. |
| `mlx.fast` Gemma engine live refresh, `f9696c9...` | Fetched the post-start head and read the new tight-grid/multitile four-bit QKV MMA, composed prefill SDPA, attention dispatcher and ragged two-pass decode implementations. The unchanged MIT licence hash is recorded. | Nothing new; the retained solution already uses exact shape dispatch, resident Q prescaling and a fused causal head-64 NAX kernel. | New paths are respectively four-bit B8/L1/K2816 projection, scale-1/GQA/head-256-or-512 fallback prefill, and L1/head-256/GQA decode. None matches fp16 dense M100000/N3072/K1024 projection or dense causal L100000/head64 attention. | Close without build or timing. Revisit only a revision with an exact dense causal head-64 full-query primitive or a relevant fp16 dense projection mechanism beyond the already swept NAX-linear tiles. |
| `mlx-train-perf`, `cdfce970...` | Acquired the clean 3.6 MB ignored checkout, recorded the MIT licence and source hashes, and read the training-only attention API, dispatch ladder, classic-SIMDgroup forward body and query-range launch budget. | Source-audit knowledge only; no code entered production. | The only measured saturated forward cell is B1/Hq32/Hkv8/N8192/D128/bf16; D64 is explicitly provisional. The project reduces forward/backward training memory and reports 5.3–5.9% lower training throughput in its measured comparisons. Its classic `simdgroup_float8x8` D64 path is not a new M5 NAX mechanism for B32/H16/N100000 inference. | Close without build or model allocation. Reconsider only with measured causal D64 M5 evidence exceeding Apple NAX/SDPA by at least 3%, or a new NAX primitive absent from production. |
| `mlx-metal-kernels`, `9fc1d38a...` | Acquired the clean 3.3 MB ignored checkout, recorded its Apache-2.0 licence and relevant hashes, and read the default, threadgroup, tiled-KV, SIMD D64 and benchmark/oracle paths. | Source-audit knowledge only; no code entered production. | The default is one thread per attention row; alternatives process one query row and sequential keys with scalar/threadgroup/SIMD reductions, no `simdgroup_matrix` or Metal-4 NAX. The checkout publishes no Apple measurements, and its fp16 oracle allows `atol=0.02`, ten times the organizer absolute tolerance. | Close without install/build/tensors/timing. Reconsider only with organizer-compatible causal D64 correctness and controlled M5 matrix/NAX evidence above the attention gate. |
| Official PyTorch nightly `2.15.0.dev20260829` | Installed its exact arm64 wheel in ignored CPython 3.13, profiled MPS SDPA dispatch, ran all ordinary rows, and tested the production NAX bridge at S8192 and S100000. | Compatibility knowledge only: PyTorch 2.15 extensions require C++20 and the retained Metal route can pass its boundaries there. | Small causal SDPA still used the MPS math backend; 39/39 trials passed but the 13-row mean was 2.142523x versus stable 2.427904x and case 3 was 0.935455x. Target absolute times were heat/order-confounded. | Retain stable PyTorch 2.8 and C++17. Revisit only a future build with a materially new MPS attention backend, behind a full-suite gate. |
| `ComfyUI-SolAttn-MPS`, `4507112...` | Read its block-summary/routing source and compiled the unmodified Metal library under the isolated PyTorch 2.15 nightly. | Source-audit knowledge only; no code entered production. | Its acceleration is approximate sparse block routing, the public dispatcher falls back for masks, and the kernel has no causal path. Its BQ32/BQ64, BK64, shared-staging core overlaps locally rejected designs. | Close for the dense causal challenge. Reconsider only if upstream publishes an exact causal route with a materially new M5 primitive. |
| `flash_attn_metal_cpp`, `f91c9b1...` | Read the scalar/vector causal and optimized kernels, verifier, build files and repository metadata. | Nothing. | No licence file; the optimized kernel documents causal failures; its causal verifier allows 0.06 maximum error; the safe kernel uses scalar/`float4` operations aimed at M1-M3. | Do not build, reuse or benchmark without a licence and a source-level reason to expect a gain over retained M5 NAX. |
| NVIDIA Model Optimizer NVFP4, official `0.46` documentation | Read the official NVFP4 recipe matrix, accuracy-risk guidance, Blackwell hardware boundary and TensorRT deployment contract after a participant-supplied social-media screenshot. No model, checkpoint, package or repository was downloaded. Primary pages: <https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/ptq.md> and <https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md>. | Methodology only: quantize selectively, keep sensitive attention projections at higher precision, and require end-to-end evaluation. That discipline already governs the retained fp16-internal/fp32-I/O routes. | NVFP4 W4A4 acceleration requires Blackwell-class NVIDIA hardware and its TensorRT ecosystem; the M5 Pro has no NVFP4 primitive. Row 14 has only 12,605,440 parameters (about 48.1 MiB fp32) while one float32 input or output is about 12.2 GiB, and the current profile assigns about 89.9% of time to bounded attention. Weight compression therefore attacks neither dominant memory nor runtime. Language-task accuracy retention also does not prove the organizer's elementwise tolerance. | Close without download, build or timing. Reconsider only an Apple-native low-bit matrix primitive with a credible >=3% whole-model bound and an organizer-predicate pass before timing. |
| PyTorch 2.8 TorchScript static inference APIs | Traced exact all-valid ordinary forwards, separated trace/freeze/inference-optimization stages, screened every non-fused row, then repeated winners through the real dispatcher. | One unregistered lazy graph lifecycle: exact cases 2/9/12 stop at freeze; exact case 3 adds `optimize_for_inference`; all invalidate on training/weight/device changes and fail closed. | Inference optimization lost on cases 1/2/4/5/9/11/12; freeze lost on 1/4/5 and became noise-scale on 11; broad `torch.compile` lost on 3/6/13; runtime shaders cannot be traced. A fresh post-packed case-9 transfer found trace at 0.990331x over five sessions and optimize at 0.628709x. Replacing SDPA inside the four retained graphs with traced explicit attention passed 20/20 trials but measured only 0.781941x/0.783006x/0.881829x/0.789353x on cases 2/3/9/12. | Retain only exact all-valid float32 MPS cases 2/3/9/12 with their independently measured modes and SDPA; case 9 specifically remains freeze-only. |

## Post-table retained MLX update

Milestone 148 supersedes the Apple-row statement that no NAX linear code had
entered production. Four legal geometries outside Apple's published eight-tile
list were screened at exact M100000/N3072/K1024. BM256/BN32 failed correctness;
BM32/BN512/BK256/WM1/WN8 was bit-identical, won the complete projection/layout
boundary `1.068162x` with 10/10 positive pairs, and then passed four balanced
S100000 complete-model sessions at a `1.019126x` mean with 33/36 positive
pairs. The pinned MLX linear template, direct-head store transformation and
small Metal-4 bridge are now retained under `solution/` with the same MIT
licence, byte-matched headers, asserted generator and prior-route fallback.

## Honest utilization assessment

The original direct head-major NAX packed-QKV candidate passed S8192, S32768
and four S100000 sessions but measured only `1.009190x`, below its predeclared
gate. Completing all eight upstream linear tiles did not rescue it. The later
BM32/BN512 edge geometry is a distinct, materially faster projection schedule:
its four-session complete-model mean is `1.019126x`, so it earned production
promotion. Exact commands and resources are in milestones 143, 147 and 148.

The current solution is not merely stock PyTorch: it materially uses
`triton-msl`-derived Metal code for the bounded fallback and measured fusion,
plus Apple's MLX NAX template for the fastest proven stress path,
while retaining PyTorch SDPA where it beat custom small-shape kernels. The
source audit prevented three attractive but losing/broken routes from consuming
the submission. The previously underused paths have now been exercised: the
complete MLX route repeatedly failed by a small number of elements, and the
first Philip-derived tile variant was numerically invalid. The successful MLX
lesson was lower-level: reuse its register-tiled kernel without crossing the
framework boundary. The BQ64 route passed four full-length continuity seeds and
produced a 3.164973x production paired geometric mean. A later source-backed
BQ128 query tile passed four bit-identical incremental production sessions at
1.029597x and two direct pre-NAX comparisons at 3.605275x. The later aligned-Q
route passed four sessions at 1.022838x, resident-Q prescaling passed four
complete-model sessions at 1.030260x, and Q-prescaled BQ256/BK32 passed five
sessions at 1.072758x over BQ128. The current BK48 gap candidate then passed
five S100000 complete-model sessions at 1.082579x over BK32. Further
experiments remain subject to the same correctness-first gate.

## Latest public-source refresh

At 2026-08-30 20:34 SGT, a targeted search for a new dense causal D64 Metal 4
or NAX forward primitive surfaced `mlx-flashattention-steel` `v2.62.1` as the
only close result. Its annotated tag dereferences to the already pinned and
audited `b1e7ac2586fde9e28412d81b62540f0bff281771`. Its dense M5 D64 forward
gate routes to Apple MLX SDPA; native release wins concern sparse/windowed,
neighborhood, decode or backward contracts. No new exact-row candidate exists,
so no duplicate build or timing was performed. See milestone 153.
