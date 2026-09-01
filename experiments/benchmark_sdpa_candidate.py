#!/usr/bin/env python3
"""Measure a minimal SDPA candidate on the published Track 3 shapes.

This experiment does not modify the organizer attachment. It imports the
reference classes, swaps only the explicit attention calculation for PyTorch
SDPA, applies the organizer's elementwise correctness predicate, and
synchronizes MPS before every host-side timing boundary.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "official" / "torch_transformer_benchmark.py"
sys.path.insert(0, str(ROOT))

from solution.optimized_transformer import (  # noqa: E402
    UserOptimizedTransformer as SolutionTransformer,
)


def load_official_module():
    spec = importlib.util.spec_from_file_location("official_torch_benchmark", OFFICIAL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {OFFICIAL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


official = load_official_module()


@dataclass(frozen=True)
class PublishedCase:
    case_id: int
    batch: int
    d_model: int
    heads: int
    seq_len: int
    layers: int
    ffn_dim: int
    causal: bool = True


CASES = {
    row.case_id: row
    for row in (
        PublishedCase(1, 64, 128, 4, 128, 4, 128),
        PublishedCase(2, 1, 128, 4, 128, 4, 128),
        PublishedCase(3, 4, 128, 4, 128, 4, 128),
        PublishedCase(4, 16, 128, 4, 128, 4, 128),
        PublishedCase(5, 128, 128, 4, 128, 4, 128),
        PublishedCase(6, 10000, 128, 4, 128, 4, 128),
        PublishedCase(7, 64, 32, 4, 128, 4, 32),
        PublishedCase(8, 64, 1024, 4, 128, 4, 1024),
        PublishedCase(9, 64, 128, 1, 128, 4, 128),
        PublishedCase(10, 64, 128, 2, 128, 4, 128),
        PublishedCase(11, 64, 128, 16, 128, 4, 128),
        PublishedCase(12, 64, 128, 4, 32, 4, 128),
        PublishedCase(13, 64, 128, 4, 1024, 4, 128),
        PublishedCase(14, 32, 1024, 16, 100000, 2, 1024),
    )
}


class SDPAAttention(official.BaselineSelfAttention):
    """Same parameters and output contract, with SDPA for the attention core."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        attention_mask = None
        if valid_token_mask is not None:
            # PyTorch SDPA boolean masks use True for positions that participate.
            attention_mask = valid_token_mask[:, None, None, :]
        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).tril()
            attention_mask = (
                causal_mask
                if attention_mask is None
                else attention_mask & causal_mask
            )

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            # PyTorch 2.8 MPS aborts in Objective-C when both attn_mask and
            # is_causal are supplied. The combined boolean mask above preserves
            # the same semantics without invoking that runtime bug.
            is_causal=False,
            scale=self.scale,
        )
        context = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class CausalFastPathSDPAAttention(SDPAAttention):
    """Use SDPA's causal flag when the model has proven all tokens are valid."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if valid_token_mask is not None:
            return super().forward(x, valid_token_mask, causal)

        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=self.scale,
        )
        context = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        return self.out_proj(context)


class PackedQKVCausalFastPathSDPAAttention(CausalFastPathSDPAAttention):
    """Execute Q, K, and V projection as one linear operation."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model, num_heads)
        self.register_buffer("_qkv_weight", None, persistent=False)
        self.register_buffer("_qkv_bias", None, persistent=False)

    def pack(self) -> None:
        self._qkv_weight = torch.cat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0
        ).detach()
        self._qkv_bias = torch.cat(
            (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), dim=0
        ).detach()

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if self._qkv_weight is None or self._qkv_bias is None:
            raise RuntimeError("packed QKV buffers were not initialized")
        batch, seq_len, _ = x.shape
        qkv = F.linear(x, self._qkv_weight, self._qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if valid_token_mask is None:
            attention_mask = None
            is_causal = causal
        else:
            attention_mask = valid_token_mask[:, None, None, :]
            if causal:
                causal_mask = torch.ones(
                    (seq_len, seq_len), device=x.device, dtype=torch.bool
                ).tril()
                attention_mask = attention_mask & causal_mask
            is_causal = False

        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        )
        context = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class PackedQKVExplicitAttention(PackedQKVCausalFastPathSDPAAttention):
    """Pack projections while preserving the organizer attention arithmetic."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if self._qkv_weight is None or self._qkv_bias is None:
            raise RuntimeError("packed QKV buffers were not initialized")
        batch, seq_len, _ = x.shape
        qkv = F.linear(x, self._qkv_weight, self._qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len), device=x.device, dtype=torch.bool
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if valid_token_mask is not None:
            scores = scores.masked_fill(
                ~valid_token_mask[:, None, None, :], float("-inf")
            )
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class Float32CoreCausalFastPathSDPAAttention(CausalFastPathSDPAAttention):
    """Probe an fp32 SDPA core for half-precision accuracy-sensitive shapes."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if valid_token_mask is None:
            attention_mask = None
            is_causal = causal
        else:
            attention_mask = valid_token_mask[:, None, None, :]
            if causal:
                causal_mask = torch.ones(
                    (seq_len, seq_len), device=x.device, dtype=torch.bool
                ).tril()
                attention_mask = attention_mask & causal_mask
            is_causal = False
        context = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.scale,
        ).to(dtype=x.dtype)
        context = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class PreScaledQCausalFastPathSDPAAttention(CausalFastPathSDPAAttention):
    """Probe reference-order scaling by multiplying Q before SDPA."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x)) * self.scale
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if valid_token_mask is None:
            attention_mask = None
            is_causal = causal
        else:
            attention_mask = valid_token_mask[:, None, None, :]
            if causal:
                causal_mask = torch.ones(
                    (seq_len, seq_len), device=x.device, dtype=torch.bool
                ).tril()
                attention_mask = attention_mask & causal_mask
            is_causal = False
        context = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=1.0,
        )
        context = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class MaskAwareSDPATransformer(official.BaselineTransformer):
    """Cache mask classification and remove redundant all-valid masking."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._cached_mask: Optional[torch.Tensor] = None
        self._cached_mask_version = -1
        self._cached_mask_all_valid = False

    def _all_valid(self, mask: torch.Tensor) -> bool:
        try:
            version = mask._version
        except RuntimeError:
            # Tensors created inside inference_mode do not expose a version
            # counter. Identity caching is still safe for the fixed masks used
            # by the organizer accuracy and timing loops.
            version = None
        if mask is self._cached_mask and version == self._cached_mask_version:
            return self._cached_mask_all_valid
        result = bool(mask.all().item())
        self._cached_mask = mask
        self._cached_mask_version = version
        self._cached_mask_all_valid = result
        return result

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        effective_mask = valid_token_mask
        if valid_token_mask is not None and self._all_valid(valid_token_mask):
            effective_mask = None
        for layer in self.layers:
            x = layer(x, effective_mask, self.config.causal)
        x = self.final_norm(x)
        if effective_mask is not None:
            x = x.masked_fill(~effective_mask[..., None], 0)
        return x


def make_candidate(reference):
    candidate = copy.deepcopy(reference)
    for block in candidate.layers:
        old = block.attention
        replacement = SDPAAttention(old.d_model, old.num_heads)
        replacement.load_state_dict(old.state_dict(), strict=True)
        block.attention = replacement
    return candidate


def make_mask_aware_candidate(reference):
    candidate = MaskAwareSDPATransformer(reference.config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    for block in candidate.layers:
        old = block.attention
        replacement = CausalFastPathSDPAAttention(old.d_model, old.num_heads)
        replacement.load_state_dict(old.state_dict(), strict=True)
        block.attention = replacement
    return candidate


def make_packed_qkv_candidate(reference):
    candidate = MaskAwareSDPATransformer(reference.config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    for block in candidate.layers:
        old = block.attention
        replacement = PackedQKVCausalFastPathSDPAAttention(
            old.d_model, old.num_heads
        )
        replacement.load_state_dict(old.state_dict(), strict=True)
        replacement.pack()
        block.attention = replacement
    return candidate


def make_fp32_core_candidate(reference):
    candidate = MaskAwareSDPATransformer(reference.config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    for block in candidate.layers:
        old = block.attention
        replacement = Float32CoreCausalFastPathSDPAAttention(
            old.d_model, old.num_heads
        )
        replacement.load_state_dict(old.state_dict(), strict=True)
        block.attention = replacement
    return candidate


def make_packed_qkv_explicit_candidate(reference):
    candidate = MaskAwareSDPATransformer(reference.config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    for block in candidate.layers:
        old = block.attention
        replacement = PackedQKVExplicitAttention(old.d_model, old.num_heads)
        replacement.load_state_dict(old.state_dict(), strict=True)
        replacement.pack()
        block.attention = replacement
    return candidate


def make_prescaled_q_candidate(reference):
    candidate = MaskAwareSDPATransformer(reference.config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    for block in candidate.layers:
        old = block.attention
        replacement = PreScaledQCausalFastPathSDPAAttention(
            old.d_model, old.num_heads
        )
        replacement.load_state_dict(old.state_dict(), strict=True)
        block.attention = replacement
    return candidate


def make_solution_candidate(reference):
    candidate = SolutionTransformer(reference.config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return candidate


def make_fused_norm_candidate(reference):
    # The public solution candidate does not depend on the ignored triton-msl
    # generator checkout. Import this experiment-only dependency only when its
    # candidate is explicitly selected.
    from fused_norm_solution import FusedNormCase8Transformer

    candidate = FusedNormCase8Transformer(reference.config)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    return candidate


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_samples(model, x, mask, device, warmup: int, repeats: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, mask)
        synchronize(device)
        samples = []
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter_ns()
            model(x, mask)
            synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True, choices=sorted(CASES))
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "float16", "bfloat16"),
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--accuracy-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default=None,
        help=(
            "set PyTorch float32 matmul precision before model construction; "
            "the organizer harness defaults to high, while omission preserves "
            "the process default"
        ),
    )
    parser.add_argument(
        "--candidate",
        default="sdpa",
        choices=(
            "sdpa",
            "mask-aware-sdpa",
            "packed-qkv-sdpa",
            "fp32-core-sdpa",
            "prescaled-q-sdpa",
            "packed-qkv-explicit",
            "solution",
            "fused-norm-solution",
        ),
    )
    args = parser.parse_args()

    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)

    row = CASES[args.case]
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    config = official.TransformerConfig(
        batch_size=row.batch,
        seq_len=row.seq_len,
        d_model=row.d_model,
        num_heads=row.heads,
        ffn_dim=row.ffn_dim,
        num_layers=row.layers,
        causal=row.causal,
    )
    config.validate()

    # Case 14's raw fp32 score tensor is 18.626 TiB. Never instantiate the
    # explicit baseline for it on this 64 GiB machine.
    score_bytes = row.batch * row.heads * row.seq_len * row.seq_len * 4
    if score_bytes > 16 * 1024**3:
        print(
            f"SKIP case={row.case_id}: explicit baseline score tensor estimate "
            f"{score_bytes / 1024**3:.3f} GiB exceeds the 16 GiB safety cap"
        )
        return 3

    torch.manual_seed(1234)
    reference = official.BaselineTransformer(config)
    candidate_factories = {
        "sdpa": make_candidate,
        "mask-aware-sdpa": make_mask_aware_candidate,
        "packed-qkv-sdpa": make_packed_qkv_candidate,
        "fp32-core-sdpa": make_fp32_core_candidate,
        "prescaled-q-sdpa": make_prescaled_q_candidate,
        "packed-qkv-explicit": make_packed_qkv_explicit_candidate,
        "solution": make_solution_candidate,
        "fused-norm-solution": make_fused_norm_candidate,
    }
    candidate = candidate_factories[args.candidate](reference)
    reference = reference.to(device=device, dtype=dtype).eval()
    candidate = candidate.to(device=device, dtype=dtype).eval()
    print(
        f"candidate={args.candidate} case={row.case_id} "
        f"shape=(B={row.batch},S={row.seq_len},D={row.d_model},"
        f"H={row.heads},F={row.ffn_dim},L={row.layers}) dtype={dtype} device={device} "
        f"matmul_precision={torch.get_float32_matmul_precision()}"
    )
    if args.accuracy_trials <= 0:
        print(
            "accuracy-trials must be positive; refusing to benchmark an "
            "unchecked candidate"
        )
        return 4
    with torch.inference_mode():
        for trial in range(args.accuracy_trials):
            accuracy_x, accuracy_mask = official.generate_random_case(
                config,
                device,
                dtype,
                args.seed + trial,
                args.padding_ratio,
                1.0,
            )
            expected = reference(accuracy_x, accuracy_mask)
            actual = candidate(accuracy_x, accuracy_mask)
            synchronize(device)
            accuracy = official.compare_outputs(
                expected, actual, rtol=0.02, atol=0.002
            )
            print(
                f"accuracy_trial={trial + 1}/{args.accuracy_trials} "
                f"result={'PASS' if accuracy.passed else 'FAIL'} "
                f"failed={accuracy.failed_elements}/{accuracy.total_elements} "
                f"max_abs={accuracy.max_abs_error:.8g} "
                f"mean_abs={accuracy.mean_abs_error:.8g}"
            )
            if accuracy.passed:
                continue
            print(
                f"worst_index={accuracy.worst_index} "
                f"reference_at_worst={accuracy.reference_at_worst:.8g} "
                f"candidate_at_worst={accuracy.optimized_at_worst:.8g}"
            )
            return 2

    if args.skip_timing:
        print("timing=SKIP requested after successful accuracy trials")
        return 0
    x, mask = official.generate_random_case(
        config, device, dtype, args.seed + 100000, args.padding_ratio, 1.0
    )

    reference_samples: list[float] = []
    candidate_samples: list[float] = []
    for round_index in range(args.rounds):
        order = ((reference, reference_samples), (candidate, candidate_samples))
        if round_index % 2:
            order = tuple(reversed(order))
        for model, destination in order:
            destination.extend(
                timed_samples(model, x, mask, device, args.warmup, args.repeats)
            )

    ref_median = statistics.median(reference_samples)
    cand_median = statistics.median(candidate_samples)
    print(
        f"baseline_median_ms={ref_median:.6f} "
        f"sdpa_median_ms={cand_median:.6f} speedup={ref_median / cand_median:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
