# Numerical precision and dtype contract

Snapshot: **2026-08-26 SGT**

This preparation-only audit describes the precision behavior of the two
unmodified organizer harnesses and records tiny baseline-only CPU smoke runs.
It does not propose reduced-precision tactics, modify either attachment, or
implement participant code.

## Declared modes and defaults

Both command-line interfaces accept `float32`, `float16`, and `bfloat16`, but
their defaults differ:

| Harness | Default dtype | Other declared dtypes | Default float32 math control |
| --- | --- | --- | --- |
| PyTorch | `float32` | `float16`, `bfloat16` | `--matmul-precision high`, `--allow-tf32` |
| TensorFlow | `float16` | `float32`, `bfloat16` | `--allow-tf32` |

Both issue a warning, rather than rejecting the run, when `float16` is selected
on CPU. Neither validates that every operator is supported for the selected
device/dtype before constructing and running the models.

## Reference computation path

PyTorch constructs the baseline and participant placeholder in float32, copies
the baseline `state_dict`, and then converts both complete models to the
requested device and dtype. Random inputs are generated directly in that
dtype. Linear projections, LayerNorm, GELU, attention scores, residuals, and
the final output therefore follow the requested tensor/model dtype, subject to
the selected backend's own internal accumulation rules.

TensorFlow constructs every Dense and LayerNormalization layer with the
requested dtype, builds both models using a tiny input of that dtype, then
copies the baseline weights. Random inputs are also generated directly in the
requested dtype. The harness does not install a separate Keras mixed-precision
policy. Internal accumulation behavior not explicitly controlled in the
organizer code remains backend- and operator-dependent.

Both references explicitly make only the attention softmax a mixed-precision
island:

```text
requested-dtype scores
-> cast to float32
-> softmax along the key dimension
-> cast probabilities back to the requested dtype
-> multiply probabilities by V
```

Thus selecting `float16` or `bfloat16` does not make the reference attention
softmax itself reduced precision. The QK score matmul and probability-times-V
matmul still use the requested dtype. The output dtype remains the requested
dtype in the untouched reference models.

## Float32 and TF32 controls

The PyTorch harness always calls
`torch.set_float32_matmul_precision(args.matmul_precision)`, whose default is
`high`. On CUDA it then explicitly sets both
`torch.backends.cuda.matmul.allow_tf32` and
`torch.backends.cudnn.allow_tf32` from `--allow-tf32`, which defaults to true.
PyTorch documents that `high` may use TF32 or a bfloat16-based internal matmul
algorithm when a fast implementation is available, without changing the
float32 output dtype. PyTorch 2.8 also documents that this setting currently
affects CUDA, so it did not change the local CPU smoke runs.

TensorFlow calls
`tf.config.experimental.enable_tensor_float_32_execution(args.allow_tf32)`;
the flag also defaults to true. TensorFlow documents that this affects selected
float32 operations only on NVIDIA Ampere-or-newer GPUs, can round supported
operation inputs from 23 to 10 precision bits, retains float32 accumulation and
dynamic range, and may not apply to every shape. It had no effect on the local
Apple CPU.

Consequently, `dtype=float32` does not by itself prove full-precision internal
matrix multiplication on a supported NVIDIA GPU. A reproducible numerical or
performance result must also state both frameworks' TF32/matmul settings and
the exact hardware.

PyTorch also documents MPS-specific process controls for fast math and for
preferring Metal matmul kernels over MPSGraph. Fresh-process screens on exact
cases 3 and 6 found no material candidate benefit; they are not part of the
retained run contract. Exact commands and results are in
[`SOLUTION_MILESTONE_89_2026-08-30.md`](SOLUTION_MILESTONE_89_2026-08-30.md).

The solution measurement harness originally inherited PyTorch's `highest`
process default even though the organizer harness explicitly defaults to
`high`. A post-start alignment screen therefore named all three supported
settings in fresh processes. Fixed-seed correctness summaries were unchanged;
`medium` raised absolute latency on the large probes and was rejected. One
complete cases-1–13 `high` point passed 39/39 comparisons at 2.513080x, and an
exact B32/S100000 `high` run completed all 3,276,800,000 outputs finite in
98.588985 seconds. All future authoritative commands explicitly name `high`.
See
[`SOLUTION_MILESTONE_126_2026-08-30.md`](SOLUTION_MILESTONE_126_2026-08-30.md).

The four retained static production graphs were then compared directly with
their current eager routes under explicit `high`. All 28 organizer comparisons
and four continuity checks passed; cases 2/3/9/12 remained faster at
1.054495x/1.201002x/1.044573x/1.172183x. This isolates the graph decisions
under the final precision contract rather than relying only on a pooled suite
point. See
[`SOLUTION_MILESTONE_128_2026-08-30.md`](SOLUTION_MILESTONE_128_2026-08-30.md).

## What the correctness gate enforces

Both harnesses cast baseline and participant outputs to float32 before applying
their elementwise finite/error predicates. The underlying output precision has
already been lost or retained by that point; this comparison cast cannot
recover precision.

PyTorch checks output shape and prints only a warning for output-dtype mismatch.
TensorFlow checks output shape but performs no output-dtype equality check.
Neither harness changes its tolerances by requested dtype. Therefore the
observable numerical contract is shape plus the fixed float32 comparison
predicate, not an exact output-dtype requirement. Exact predicate and boundary
behavior are documented separately in
[`CORRECTNESS_ORACLE.md`](CORRECTNESS_ORACLE.md).

## Tiny unmodified CPU dtype matrix

All runs used the official scripts without modification, the untouched
participant fallback (which calls the baseline), and one tiny shape:

```text
B=1, S=4, D=8, H=2, F=16, L=1, non-causal, padding=0
accuracy trials=1, warm-up=1, repeats=2, rounds=1
```

Exact PyTorch commands:

```bash
/usr/bin/time -l .venv/bin/python official/torch_transformer_benchmark.py --device cpu --batch-size 1 --seq-len 4 --d-model 8 --heads 2 --ffn-dim 16 --layers 1 --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1 --dtype float32
/usr/bin/time -l .venv/bin/python official/torch_transformer_benchmark.py --device cpu --batch-size 1 --seq-len 4 --d-model 8 --heads 2 --ffn-dim 16 --layers 1 --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1 --dtype float16
/usr/bin/time -l .venv/bin/python official/torch_transformer_benchmark.py --device cpu --batch-size 1 --seq-len 4 --d-model 8 --heads 2 --ffn-dim 16 --layers 1 --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1 --dtype bfloat16
```

Exact TensorFlow commands:

```bash
/usr/bin/time -l .venv/bin/python official/tensorflow_transformer_benchmark.py --device cpu --batch-sizes 1 --qkv-dims 8 --seq-lens 4 --heads 2 --ffn-dim 16 --layers 1 --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1 --max-estimated-memory-gib 1 --dtype float32 --report /tmp/track3-tf-float32-report.md
/usr/bin/time -l .venv/bin/python official/tensorflow_transformer_benchmark.py --device cpu --batch-sizes 1 --qkv-dims 8 --seq-lens 4 --heads 2 --ffn-dim 16 --layers 1 --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1 --max-estimated-memory-gib 1 --dtype float16 --report /tmp/track3-tf-float16-report.md
/usr/bin/time -l .venv/bin/python official/tensorflow_transformer_benchmark.py --device cpu --batch-sizes 1 --qkv-dims 8 --seq-lens 4 --heads 2 --ffn-dim 16 --layers 1 --accuracy-trials 1 --warmup 1 --repeats 2 --benchmark-rounds 1 --max-estimated-memory-gib 1 --dtype bfloat16 --report /tmp/track3-tf-bfloat16-report.md
```

Observed results:

| Framework | Dtype | Exit | Accuracy | Baseline median | Placeholder median | Real time | Maximum RSS |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| PyTorch 2.8.0 | float32 | 0 | PASS, `0/32` failed | 0.1821 ms | 0.2972 ms | 0.64 s | 202,326,016 B |
| PyTorch 2.8.0 | float16 | 0 | PASS, `0/32` failed | 0.3861 ms | 0.3251 ms | 0.63 s | 202,440,704 B |
| PyTorch 2.8.0 | bfloat16 | 0 | PASS, `0/32` failed | 0.6825 ms | 0.5767 ms | 0.62 s | 202,539,008 B |
| TensorFlow 2.20.0 | float32 | 0 | PASS, max errors `0` | 0.2493 ms | 0.2589 ms | 2.28 s | 434,028,544 B |
| TensorFlow 2.20.0 | float16 | 0 | PASS, max errors `0` | 0.2494 ms | 0.2561 ms | 2.23 s | 436,289,536 B |
| TensorFlow 2.20.0 | bfloat16 | 0 | PASS, max errors `0` | 0.2523 ms | 0.2586 ms | 2.20 s | 436,256,768 B |

All six runs reported zero swaps and zero block-output operations. PyTorch and
TensorFlow printed their documented float16-on-CPU warning. TensorFlow also
printed the already-recorded urllib3/LibreSSL compatibility warning; it did not
affect the successful runs. Its reports were written under `/private/tmp` and
were not added to the repository.

The participant placeholder is the same baseline implementation with copied
weights, so zero error is expected. With only two timing samples, the displayed
latencies and speedups are startup/order noise. In addition, the six commands
were launched concurrently, so they competed for host resources. Their latency
figures are **not** accepted as meaningful framework, dtype, or optimization
comparisons. These runs establish only that every declared dtype completed this
tiny CPU code path on this host; they do not establish target-GPU support,
full-shape coverage, or competition performance.

## Primary sources

- PyTorch 2.8 `set_float32_matmul_precision`:
  <https://docs.pytorch.org/docs/2.8/generated/torch.set_float32_matmul_precision.html>
- PyTorch 2.8 TF32 and reduced-precision CUDA semantics:
  <https://docs.pytorch.org/docs/2.8/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices>
- PyTorch MPS environment variables:
  <https://docs.pytorch.org/docs/stable/mps_environment_variables.html>
- TensorFlow TF32 control:
  <https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_tensor_float_32_execution>
- TensorFlow softmax type behavior:
  <https://www.tensorflow.org/api_docs/python/tf/nn/softmax>
