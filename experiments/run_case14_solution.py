#!/usr/bin/env python3
"""Run the bounded solution on the published Track 3 stress shape.

This intentionally does not instantiate the organizer baseline: its explicit
float32 attention score tensor alone would require 18.626 TiB at batch 32.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import resource
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution.optimized_transformer import UserOptimizedTransformer  # noqa: E402


def load_official_module():
    path = ROOT / "official" / "torch_transformer_benchmark.py"
    spec = importlib.util.spec_from_file_location("official_case14_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1401)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default=None,
        help=(
            "set PyTorch float32 matmul precision before model construction; "
            "the organizer harness defaults to high"
        ),
    )
    parser.add_argument(
        "--report-item-timings",
        action="store_true",
        help=(
            "print synchronized per-batch-item latency and allocation diagnostics; "
            "the diagnostics run after each item's measured interval"
        ),
    )
    parser.add_argument(
        "--warmup-items",
        type=int,
        default=0,
        help=(
            "run this many synchronized one-item forwards before the measured "
            "batch loop; zero preserves the historical exact-run protocol"
        ),
    )
    parser.add_argument(
        "--nax-command-buffer-timestamps",
        action="store_true",
        help=(
            "compile an experiment-only NAX bridge that attaches public Metal "
            "GPU timestamp handlers at each native attention dispatch"
        ),
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.seq_len <= 0:
        raise ValueError("batch-size and seq-len must be positive")
    if args.warmup_items < 0:
        raise ValueError("warmup-items must be non-negative")
    if not torch.backends.mps.is_available():
        raise RuntimeError("this stress runner requires Apple MPS")
    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)

    official = load_official_module()
    config = official.TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    config.validate()
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed - 1)
    model = UserOptimizedTransformer(config).to(
        device="mps", dtype=dtype
    ).eval()
    nax_command_buffer_probe = None
    if args.nax_command_buffer_timestamps:
        from experiments.nax_command_buffer_probe import (
            install_nax_command_buffer_probe,
        )

        nax_command_buffer_probe = install_nax_command_buffer_probe()
    x, mask = official.generate_random_case(
        config,
        torch.device("mps"),
        dtype,
        args.seed,
        0.0,
        1.0,
    )
    # Never apply a diagnostic operation to a view backed by the published
    # 3.2768-billion-element output. MPS may carry the oversized backing-store
    # descriptor into that operation even when the visible view is one item.
    # Instead, validate each one-item result before copying it into the global
    # output. The elapsed figure excludes these diagnostics.
    finite = True
    total_elements = 0
    total_sum = 0.0
    total_square_sum = 0.0
    max_abs = 0.0
    elapsed_ns = 0
    item_elapsed_ms = []
    nax_command_buffer_gpu_ms = []
    output = torch.empty_like(x)
    torch.mps.synchronize()
    with torch.inference_mode():
        for warmup_index in range(args.warmup_items):
            batch_index = warmup_index % args.batch_size
            chunk_mask = mask[batch_index : batch_index + 1]
            started = time.perf_counter_ns()
            warmup_output = model._forward_one_chunk(
                x[batch_index : batch_index + 1], chunk_mask
            )
            torch.mps.synchronize()
            warmup_ms = (time.perf_counter_ns() - started) / 1e6
            if args.report_item_timings or args.nax_command_buffer_timestamps:
                print(
                    f"warmup_item={warmup_index + 1} "
                    f"source_batch_item={batch_index + 1} "
                    f"elapsed_ms={warmup_ms:.6f} "
                    f"mps_current_gib="
                    f"{torch.mps.current_allocated_memory()/1024**3:.6f} "
                    f"mps_driver_gib="
                    f"{torch.mps.driver_allocated_memory()/1024**3:.6f}",
                    flush=True,
                )
            del warmup_output
        if nax_command_buffer_probe is not None:
            nax_command_buffer_probe.take_nax_telemetry()
        torch.mps.synchronize()
        for batch_index in range(args.batch_size):
            chunk_mask = mask[batch_index : batch_index + 1]
            started = time.perf_counter_ns()
            batch_output = model._forward_one_chunk(
                x[batch_index : batch_index + 1], chunk_mask
            )
            output[batch_index : batch_index + 1].copy_(batch_output)
            torch.mps.synchronize()
            item_ns = time.perf_counter_ns() - started
            elapsed_ns += item_ns
            item_elapsed_ms.append(item_ns / 1e6)

            nax_records = []
            nax_unique_gpu_ms = 0.0
            if nax_command_buffer_probe is not None:
                nax_records = list(nax_command_buffer_probe.take_nax_telemetry())
                if not nax_records:
                    raise RuntimeError("NAX command-buffer probe returned no records")
                errors = [record["error"] for record in nax_records if record["error"]]
                if errors:
                    raise RuntimeError(f"NAX command-buffer probe reported: {errors}")
                unique_buffers = {}
                for record in nax_records:
                    unique_buffers[record["buffer_id"]] = max(
                        unique_buffers.get(record["buffer_id"], 0.0),
                        record["gpu_ms"],
                    )
                nax_unique_gpu_ms = sum(unique_buffers.values())
                nax_command_buffer_gpu_ms.append(nax_unique_gpu_ms)

            if args.report_item_timings or args.nax_command_buffer_timestamps:
                item_max_rss_gib = (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
                )
                line = (
                    f"item={batch_index + 1} "
                    f"elapsed_ms={item_elapsed_ms[-1]:.6f} "
                    f"cumulative_ms={sum(item_elapsed_ms):.6f} "
                    f"mps_current_gib="
                    f"{torch.mps.current_allocated_memory()/1024**3:.6f} "
                    f"mps_driver_gib="
                    f"{torch.mps.driver_allocated_memory()/1024**3:.6f} "
                    f"process_max_rss_gib={item_max_rss_gib:.6f}"
                )
                if nax_records:
                    line += (
                        f" nax_record_count={len(nax_records)} "
                        f"nax_unique_buffer_count="
                        f"{len({record['buffer_id'] for record in nax_records})} "
                        f"nax_unique_gpu_ms={nax_unique_gpu_ms:.6f} "
                        f"host_minus_nax_gpu_ms="
                        f"{item_elapsed_ms[-1] - nax_unique_gpu_ms:.6f}"
                    )
                print(line, flush=True)

            finite = finite and bool(torch.isfinite(batch_output).all().item())
            elements = batch_output.numel()
            batch_float = batch_output.float()
            total_elements += elements
            total_sum += batch_float.mean().item() * elements
            total_square_sum += batch_float.square().mean().item() * elements
            max_abs = max(max_abs, batch_float.abs().max().item())
    elapsed_ms = elapsed_ns / 1e6
    mean = total_sum / total_elements
    variance = max(0.0, total_square_sum / total_elements - mean * mean)
    std = math.sqrt(variance)
    max_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3
    print(
        f"shape={tuple(output.shape)} dtype={output.dtype} finite={finite} "
        f"elapsed_ms={elapsed_ms:.6f} "
        f"matmul_precision={torch.get_float32_matmul_precision()}"
    )
    print(f"mean={mean:.9g} std={std:.9g} max_abs={max_abs:.9g}")
    print(
        f"mps_current_gib={torch.mps.current_allocated_memory()/1024**3:.6f} "
        f"mps_driver_gib={torch.mps.driver_allocated_memory()/1024**3:.6f} "
        f"process_max_rss_gib={max_rss_gib:.6f}"
    )
    if args.report_item_timings or args.nax_command_buffer_timestamps:
        first_count = max(1, len(item_elapsed_ms) // 4)
        first_median = statistics.median(item_elapsed_ms[:first_count])
        last_median = statistics.median(item_elapsed_ms[-first_count:])
        item_mean = statistics.fmean(item_elapsed_ms)
        centered_indices = [
            index - (len(item_elapsed_ms) - 1) / 2
            for index in range(len(item_elapsed_ms))
        ]
        slope_denominator = sum(value * value for value in centered_indices)
        slope = (
            sum(
                index * (elapsed - item_mean)
                for index, elapsed in zip(centered_indices, item_elapsed_ms)
            )
            / slope_denominator
            if slope_denominator
            else 0.0
        )
        print(
            f"item_count={len(item_elapsed_ms)} "
            f"item_min_ms={min(item_elapsed_ms):.6f} "
            f"item_median_ms={statistics.median(item_elapsed_ms):.6f} "
            f"item_max_ms={max(item_elapsed_ms):.6f} "
            f"first_quarter_median_ms={first_median:.6f} "
            f"last_quarter_median_ms={last_median:.6f} "
            f"last_over_first={last_median/first_median:.6f} "
            f"linear_slope_ms_per_item={slope:.6f}"
        )
    if nax_command_buffer_gpu_ms:
        print(
            f"nax_item_count={len(nax_command_buffer_gpu_ms)} "
            f"nax_unique_gpu_total_ms={sum(nax_command_buffer_gpu_ms):.6f} "
            f"nax_unique_gpu_median_ms="
            f"{statistics.median(nax_command_buffer_gpu_ms):.6f} "
            f"nax_unique_gpu_max_ms={max(nax_command_buffer_gpu_ms):.6f} "
            f"host_minus_nax_gpu_total_ms="
            f"{sum(item_elapsed_ms) - sum(nax_command_buffer_gpu_ms):.6f}"
        )
    return 0 if finite else 2


if __name__ == "__main__":
    raise SystemExit(main())
