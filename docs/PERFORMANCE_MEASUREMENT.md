# Performance measurement and aggregation audit

Snapshot: **2026-08-26 SGT**

This preparation-only audit describes exactly what the unmodified organizer
harnesses measure and report. It does not benchmark participant code, recommend
an optimization, or claim that any displayed statistic is the organizer's
authoritative judging score.

## Live-source recheck

The Lark document was re-read in Chrome on 2026-08-26. It still displayed
`Last updated: Aug 26`, the Track 3 section and judging table were unchanged,
and the resources section still exposed exactly two attachment controls:
`torch_transformer_benchmark.py` and
`tensorflow_transformer_benchmark.py`. Both links still resolved in the page
as `javascript:void(0);`; no stable direct attachment URL or adjacent licence
was visible.

Devpost Resources still linked only to the general information document and
still said public problem statements would be released on 27 August. Devpost
Updates still contained only its placeholder announcement. No new Track 3
target-GPU, evaluator, shape, failure-policy, or scoring clarification was
found. This was a read-only check; no organizer contact or registration action
was attempted.

## What one latency sample contains

Both harnesses measure one complete call of the six-layer Transformer stack,
not one isolated GPU kernel. Both generate one fixed input and valid-token mask
before timing, warm both models, and then reuse that input for every timed
call. The following are outside the measured interval:

- random input and mask generation;
- host-to-device placement performed by input generation;
- model construction, weight copying, and device/dtype conversion;
- correctness trials;
- runner/wrapper construction and compilation or tracing first-use cost; and
- report generation and cleanup.

PyTorch CUDA creates a start and end `torch.cuda.Event(enable_timing=True)` for
each inference, records them on the current stream, queues all calls in that
round, synchronizes the device, and then reads event elapsed times. PyTorch's
documentation defines CUDA events as synchronization markers for accurate
device timing and states that `elapsed_time` reports milliseconds between the
two recorded events. Consequently, this is device-event time, not host
wall-clock call time.

On non-CUDA PyTorch devices, each call is wrapped with
`time.perf_counter_ns()`. The harness synchronizes CUDA only. CPU calls are
synchronous, but Apple MPS dispatch is not explicitly synchronized; therefore
the existing MPS numbers remain invalid as completed-device latency.

TensorFlow globally calls
`tf.config.experimental.set_synchronous_execution(True)` and wraps each runner
call with `time.perf_counter_ns()`. TensorFlow documents that `True` executes
each operation synchronously, so the interval is synchronous host wall time,
including framework call overhead as well as completion of the work. The final
output is materialized again after each repeat group, outside all individual
sample intervals. Python defines `perf_counter` as the highest-resolution
available clock for short durations and `perf_counter_ns` as its integer
nanosecond form.

Because PyTorch CUDA uses device events while TensorFlow uses synchronous host
wall time, their absolute latency values are not the same measurement and
should not be compared as if they came from one standardized protocol.

## Sampling and statistics

| Property | PyTorch default | TensorFlow default |
| --- | ---: | ---: |
| Accuracy trials before timing | 5 | 3 per case |
| Warm-up calls per model | 20 | 10 |
| Timed rounds | 3 | 3 |
| Calls per model per round | 100 | 30 |
| Pooled samples per model | 300 | 90 |
| Displayed speedup | baseline median / participant median | baseline median / participant median |

Warm-up order is always baseline first, participant second. Timed round order
alternates, starting with the baseline:

```text
round 1: baseline, participant
round 2: participant, baseline
round 3: baseline, participant
```

Thus the default odd number of rounds is not order-balanced: baseline runs
first in two rounds, and participant code runs first in one. All samples from
all rounds are pooled; the reported speedup is the ratio of two separately
pooled medians, not a median of paired per-call or per-round speedups.

Both `TimingResult` classes implement mean, median, minimum, and a linearly
interpolated p90. PyTorch prints all four. TensorFlow retains all four in the
object but writes only medians and speedup to its Markdown table. No confidence
interval, variance, device clock, temperature, power, utilization, or measured
peak-memory statistic is collected.

By default, PyTorch skips timing and exits `2` when accuracy fails; its
`--benchmark-on-failure` flag can override only the timing skip. TensorFlow has
no equivalent override: an accuracy failure ends that case before timing and
is recorded as `FAIL`.

## Throughput and multi-case aggregation

Both harnesses calculate throughput as:

```text
B * S * 1000 / median_latency_ms
```

This is nominal sequence positions per second. It counts all `B*S` positions
even when a nonzero padding ratio makes some tokens invalid; it is not valid
tokens per second.

PyTorch runs one shape per process and has no cross-shape aggregate.
TensorFlow gives every successfully benchmarked case equal weight and reports:

```text
geometric mean = exp(mean(log(per-case speedup)))
minimum completed-case speedup
maximum completed-case speedup
```

The geometric mean is unweighted by batch size, sequence length, token count,
MAC count, or runtime. Cases marked `SKIPPED`, `OOM`, `FAIL`, or `ERROR` have no
benchmark object and are excluded. Moreover, `SKIPPED` and `OOM` do not make
the TensorFlow process exit nonzero; only `FAIL` and `ERROR` do. Therefore an
exit code of zero and a high geometric mean can coexist with incomplete shape
coverage.

The official challenge statement asks for improved performance and a technical
report, but it does not state that either script's median ratio or TensorFlow
geometric mean is the binding judging metric. Exact commands, completed-case
counts, per-shape results, framework, device, dtype, compile flags, TF32 mode,
and failure/skip rows are all required context for interpreting any reported
speedup.

## Synthetic aggregation probe

The committed probe imports the two organizer scripts and exercises only their
statistic/report functions with synthetic timing values. It does not construct
or execute a Transformer.

Command:

```bash
/usr/bin/time -l .venv/bin/python scripts/probe_timing_aggregation.py
```

Observed exit code: `0`. Selected literal stdout:

```text
torch_timing: mean=2.5 median=2.5 p90=3.7 min=1.0
tensorflow_timing: mean=2.5 median=2.5 p90=3.7 min=1.0
tf_report_assertion=- 有效组合数：`2`
tf_report_assertion=- 加速比几何平均值：`4.000x`
tf_report_assertion=- 最低加速比：`2.000x`
tf_report_assertion=- 最高加速比：`8.000x`
default_round_order=baseline-first,optimized-first,baseline-first
```

The synthetic TensorFlow input included two completed speedups (`2x`, `8x`)
plus one each of `SKIPPED`, `OOM`, `FAIL`, and `ERROR`; the asserted geometric
mean was `4x`, proving that only the two completed cases entered that summary.
Observed resource use was `3.52 s` real (`2.20 s` user, `0.49 s` system),
`547,127,296` bytes maximum resident set size, zero swaps, and zero block-output
operations. The existing urllib3/LibreSSL warning appeared on stderr and did
not affect the successful assertions.

## Primary sources

- Official Track 3 document:
  <https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf#RNYvddBmXosHGbxr9jfmxYgOydd>
- Devpost Resources: <https://tiktoktechjam2026.devpost.com/resources>
- Devpost Updates: <https://tiktoktechjam2026.devpost.com/updates>
- PyTorch 2.8 CUDA Event documentation:
  <https://docs.pytorch.org/docs/2.8/generated/torch.cuda.Event.html>
- PyTorch 2.8 CUDA synchronization documentation:
  <https://docs.pytorch.org/docs/2.8/generated/torch.cuda.synchronize.html>
- TensorFlow synchronous-execution documentation:
  <https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_synchronous_execution>
- Python 3.9 performance-counter documentation:
  <https://docs.python.org/3.9/library/time.html#time.perf_counter_ns>
