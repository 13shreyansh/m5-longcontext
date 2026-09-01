# Reproducibility, seeds, and purity assumptions

Snapshot: **2026-08-26 SGT**

This preparation-only audit traces the official seed schedules and execution
state. It uses tiny random-input generator probes only; it does not construct
or execute a Transformer, alter an organizer attachment, or design participant
code.

## PyTorch seed schedule

With the default `--seed 1234`, the harness calls `torch.manual_seed(1234)`
before constructing the baseline and participant placeholder. On CUDA it also
calls `torch.cuda.manual_seed_all(1234)`. The baseline is initialized first,
the placeholder second, and the baseline weights are then copied into the
placeholder. The two initial construction streams therefore need not have
created equal values, but the strict post-construction copy makes the untouched
models equal.

Input generation does not consume that global stream. Each case creates a new
`torch.Generator(device=device)`, seeds it with the explicit case seed, and uses
that generator for both the normal input and any random valid lengths.

Default input seeds are:

```text
accuracy trials: 1234, 1235, 1236, 1237, 1238
timing input:     101234
```

Every accuracy trial creates a new input. Timing creates one different input
once and reuses the same tensor and mask for every warm-up and timed call.

## TensorFlow seed schedule

The TensorFlow entrypoint first calls `tf.keras.utils.set_random_seed(1234)`.
For each one-based case index `i`, it resets global Python, NumPy, and
TensorFlow RNG state using:

```text
model-construction seed = 1234 + i * 1009
```

The baseline and placeholder are initialized sequentially and the baseline
weights are then copied. Inputs are independent of those stateful initializer
streams: `generate_random_case` uses `tf.random.stateless_normal` with stream
identifier `1`, and stateless uniform valid lengths with stream identifier `2`.

For case `i`, default input seeds are:

```text
accuracy trial t = 1234 + i * 100000 + t, where t = 0, 1, 2
timing input      = 1234 + i * 100000 + 100000
```

This means case 1 timing and case 2 accuracy trial 0 both use scalar seed
`201234` before the separate stream identifier is appended. Their configured
shapes differ, so this does not imply identical tensors; it does show that the
case/phase scalar-seed namespaces are not disjoint.

As in PyTorch, each accuracy trial gets a new input, while one separate timing
input is reused for all warm-ups and samples in that case.

## What is and is not deterministic

The input generators are explicitly seeded. TensorFlow's official
`stateless_normal` documentation promises repeated same-seed/same-shape output
on the same version and CPU/GPU hardware, but warns that values may change
between TensorFlow versions or on other hardware types. PyTorch's
reproducibility guide is broader: it says complete reproducibility is not
guaranteed across releases, platforms, or CPU versus GPU even with identical
seeds.

Neither organizer harness enables its framework's deterministic-operation
mode:

- no `torch.use_deterministic_algorithms(True)`, deterministic-debug mode,
  cuDNN deterministic flag, or cuBLAS reproducibility environment check; and
- no `tf.config.experimental.enable_op_determinism()`.

TensorFlow documents that operation determinism is disabled by default and
that GPU operations may produce different results with identical inputs.
PyTorch similarly documents deterministic-algorithm controls as separate from
seeding. The organizer harnesses therefore make inputs repeatable within a
fixed environment but do not establish repeatable model outputs for every
allowed backend or implementation.

The accuracy loops compare one baseline call with one participant call for
each input. They do not call either model twice on the same accuracy input and
do not test run-to-run output equality. Timing repeatedly uses one input but
does not inspect any timed output for numerical stability.

## Unenforced purity and state assumptions

Both accuracy loops pass the same input and mask objects first to the baseline,
then to the participant model. They do not clone those tensors or check them
for mutation afterward. The same timing tensors are reused across all calls,
including alternating baseline/participant order.

Both models also continue from correctness trials into warm-up and timing
without restoring weights, buffers, variables, or other state. PyTorch uses
`eval()` and `inference_mode()`; TensorFlow passes `training=False`; the
organizer reference itself is stateless during inference. These settings do
not prove that an arbitrary participant implementation is side-effect free.

Accordingly, the local evaluator assumes—but does not enforce—that forward
calls do not mutate inputs, masks, parameters, buffers, variables, or other
state in a way that affects later calls. A hidden evaluator's purity,
repeatability, and nondeterminism requirements remain unspecified.

## Tiny generator probe

The committed probe uses only `[3,8,4]` float32 CPU inputs with seed `777`,
padding ratio `0.5`, and input scale `1.25`. It calls each official generator
twice with the same seed and once with a different seed.

Command:

```bash
/usr/bin/time -l .venv/bin/python scripts/probe_case_generation.py
```

Observed exit code: `0`. Literal stdout:

```text
torch_same_seed_repeatable=True x_sha256=c42c8a4df9f2a4f33a69c13b00c9b59f4fe13c07ad314b16ddb8f380cf3b9b43
torch_valid_lengths=[7, 8, 5] mask_sha256=87f887976fde2c68a45bc6c112eae364e2705935fbccd126de145b3fbf1b0c61
tensorflow_same_seed_repeatable=True x_sha256=b38d7056b62a792ea09565acac2cc7d396accc8c799d9da52e6875a9337bdbd9
tensorflow_valid_lengths=[7, 7, 5] mask_sha256=15b2be26b8a3013c9d82988e467d7de8a7fde5407e2ee40ce79ef11c1cd38b3d
same_numeric_seed_cross_framework_input_equal=False
torch_default_accuracy_seeds=[1234, 1235, 1236, 1237, 1238]
torch_default_timing_seed=101234
tensorflow_case1_model_seed=2243
tensorflow_case1_accuracy_seeds=[101234, 101235, 101236]
tensorflow_case1_timing_seed=201234
tensorflow_case2_model_seed=3252
tensorflow_case2_accuracy_seeds=[201234, 201235, 201236]
tensorflow_case1_timing_equals_case2_accuracy0=True
```

Same-seed inputs and masks were byte-identical within each installed framework;
different-seed inputs differed. The same numeric seed produced different
PyTorch and TensorFlow values and padding lengths, confirming that the two
harnesses do not share a cross-framework test corpus.

Observed resource use was `2.28 s` real (`2.01 s` user, `0.24 s` system),
`559,431,680` bytes maximum resident set size, zero swaps, and zero block-output
operations. The existing urllib3/LibreSSL warning appeared on stderr and did
not affect the successful assertions.

These hashes are evidence only for the installed PyTorch 2.8.0, TensorFlow
2.20.0, Python 3.9.6, and Apple CPU environment. They are not claimed as
portable golden vectors.

## Primary sources

- PyTorch 2.8 reproducibility guide:
  <https://docs.pytorch.org/docs/2.8/notes/randomness.html>
- PyTorch deterministic-algorithm API:
  <https://docs.pytorch.org/docs/2.8/generated/torch.use_deterministic_algorithms.html>
- TensorFlow stateless normal RNG:
  <https://www.tensorflow.org/api_docs/python/tf/random/stateless_normal>
- TensorFlow operation determinism:
  <https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism>

