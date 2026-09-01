# Solution milestone 84 — PyTorch nightly environment rejected

Date: 2026-08-30 SGT

## Outcome

An isolated official PyTorch nightly was tested because a newer MPS runtime
could improve the organizer reference, the retained PyTorch operations, or the
custom Metal bridge without a source change. It did not expose a streaming MPS
SDPA backend: a profiled small causal call still selected
`aten::_scaled_dot_product_attention_math_for_mps`. Every one of the 39 fresh
float32 accuracy trials over published cases 1–13 passed, and the production
Metal route also compiled and passed its S8192 boundaries after a temporary
C++20 compatibility probe. However, the nightly's complete 13-row geometric
mean was only **2.142523x**, versus the stable champion's **2.427904x**, and
case 3 regressed to **0.935455x**. The nightly environment is rejected; no
production source, dependency pin, or dispatcher changed.

## Isolated environment and provenance

The environment, uv cache, downloaded wheel, and native build products live
only under ignored `artifacts/` paths. No secrets were used or stored.

```bash
export UV_CACHE_DIR="$PWD/artifacts/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/artifacts/uv-python"
nightly_root="$PWD/artifacts/nightly-pytorch-mps"
artifacts/tools/uv-bin/uv python install 3.13
artifacts/tools/uv-bin/uv venv --python 3.13 "$nightly_root/.venv"
artifacts/tools/uv-bin/uv pip install \
  --python "$nightly_root/.venv/bin/python" \
  --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
artifacts/tools/uv-bin/uv pip install \
  --python "$nightly_root/.venv/bin/python" numpy ninja
```

Observed runtime:

- CPython `3.13.15`, Apple Clang `22.1.3`;
- PyTorch `2.15.0.dev20260829`, Git revision
  `332a69317e22b105a867838624a87984e05021e2`;
- MPS built and available;
- NumPy `2.5.2` and Ninja `1.13.0`;
- isolated environment `658 MiB`, uv cache `2.8 GiB`, uv Python `70 MiB`;
- wheel URL:
  `https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.15.0.dev20260829-cp313-cp313-macosx_14_0_arm64.whl`;
- wheel size `127,472,326` bytes; SHA-256
  `eae6011100e688198c075f241d1059411ed3778f82808c9ca38e1fea29366fee`.
- wheel metadata licence expression `Apache-2.0 AND Apache-2.0 WITH
  LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT`;
  bundled top-level licence SHA-256
  `bd018feef8825e88181c84eb7e3aa4eafb8f08a20d9fd6ef948569610c4a3e43`.

## Complete ordinary-row gate

Each nightly row used the same current source, seed 7903, three independent
accuracy trials, two warmups, three timed samples per round, and three
alternating rounds:

```bash
for case_id in $(seq 1 13); do
  artifacts/nightly-pytorch-mps/.venv/bin/python \
    experiments/benchmark_sdpa_candidate.py \
    --case "$case_id" --candidate solution --device mps --dtype float32 \
    --seed 7903 --accuracy-trials 3 --warmup 2 --repeats 3 --rounds 3
done
```

| Case | Baseline median ms | Candidate median ms | Speedup |
|---:|---:|---:|---:|
| 1 | 7.411167 | 2.418292 | 3.064629x |
| 2 | 2.653792 | 1.431291 | 1.854125x |
| 3 | 1.591833 | 1.701667 | 0.935455x |
| 4 | 2.188208 | 1.145959 | 1.909499x |
| 5 | 15.020916 | 4.390000 | 3.421621x |
| 6 | 1746.489500 | 529.450417 | 3.298684x |
| 7 | 5.298125 | 3.600750 | 1.471395x |
| 8 | 91.155750 | 28.853375 | 3.159275x |
| 9 | 4.562875 | 3.499833 | 1.303741x |
| 10 | 5.671541 | 2.675583 | 2.119740x |
| 11 | 20.883125 | 14.786375 | 1.412322x |
| 12 | 1.786833 | 1.235584 | 1.446144x |
| 13 | 334.274000 | 50.123125 | 6.669057x |

All **39/39** accuracy trials passed. The geometric mean was **2.142523x**.
This is not mixed with the stable suite because framework changes alter both
baseline and candidate timings. The like-for-like stable champion remains the
better submitted environment at **2.427904x**.

## Stress-route compatibility and confounding

PyTorch 2.15 requires C++20 for extensions, while the retained PyTorch 2.8
bridge intentionally compiles as C++17. A temporary version-gated C++20 probe
was used only inside the experiment. The nightly bridge then compiled and the
production safe-BK48 S8192 run passed:

- champion continuity: `0/8,388,608` failures, max absolute
  `0.000970005989`, mean absolute `0.0000521900365`;
- explicit reference: `0/8,388,608` failures, max absolute
  `0.000939488411`, mean absolute `0.000112965085`.

Two target-length order reversals also passed correctness. In the first order,
stable BK48/BK32 was `1.079113x` and nightly was `1.011006x`. In the reversed
order, nightly was `1.162695x` and stable was `1.174918x`. Absolute medians
ranged from 1.766 to 4.407 seconds as the machine heated, despite no reported
macOS thermal/performance warning. They prove nightly compatibility, not a
nightly speed advantage. The temporary C++20 edit was reverted completely.

After the rejection, all organizer attachment hashes, both executable source-
provenance chains and all **59/59 tests** passed under the retained stable
environment. `git diff` contains documentation only; production source is
unchanged from milestone 83.

## Whole-suite scorecard after milestone 84

| Published rows | Verified current evidence | Boundary |
|---|---|---|
| 1–13 | Stable PyTorch 2.8 champion: 2.427904x synchronized float32 geometric mean; nightly 2.15 candidate rejected at 2.142523x despite 39/39 passing trials | Each framework is compared only within its own same-session baseline/candidate measurements |
| 14 | Bounds-safe BK48; five-session BK48-over-BK32 mean 1.082579x; current exact B32 97.173 s | Full explicit target reference remains physically infeasible; nightly target timings were order/thermal-confounded and not promoted |

Official organizer score and MFU remain **unknown**. No submission, push,
publication, organizer contact or visibility action occurred.
