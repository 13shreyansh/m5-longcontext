"""Experimental case-8 residual-plus-LayerNorm integration.

The MSL generator is imported from the pinned ignored MIT-licensed triton-msl
checkout. This module is not the promoted solution and is not standalone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts" / "upstreams" / "triton-msl"))

from solution.optimized_transformer import UserOptimizedTransformer  # noqa: E402
from triton_msl.codegen._msl_templates import (  # noqa: E402
    make_fused_residual_norm_kernel,
)


class FusedNormCase8Transformer(UserOptimizedTransformer):
    """Fuse residual additions with the following LayerNorm for case 8."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._fused_library = None
        self._fused_buffers = None

    def _eligible(self, x, valid_token_mask) -> bool:
        return (
            x.device.type == "mps"
            and x.dtype == torch.float32
            and x.shape[-1] == 1024
            and self.config.ffn_dim == 1024
            and valid_token_mask is not None
            and self._all_valid(valid_token_mask)
        )

    def _ensure_fused_runtime(self, x) -> None:
        if self._fused_library is None:
            source = make_fused_residual_norm_kernel(
                block_size=512, dtype="fp32", eps=1e-5
            )
            self._fused_library = torch.mps.compile_shader(source)
        shape_key = (tuple(x.shape), x.dtype, x.device)
        if self._fused_buffers is None or self._fused_buffers[0] != shape_key:
            slots = tuple(
                (torch.empty_like(x), torch.empty_like(x)) for _ in range(2)
            )
            self._fused_buffers = (shape_key, slots)

    def _fused_residual_norm(self, x, residual, norm, slot_index):
        _, slots = self._fused_buffers
        normalized, residual_output = slots[slot_index]
        rows = x.numel() // x.shape[-1]
        self._fused_library.fused_residual_norm(
            x,
            residual,
            norm.weight,
            norm.bias,
            normalized,
            residual_output,
            x.shape[-1],
            threads=rows * 512,
            group_size=512,
        )
        return residual_output, normalized

    def forward(self, x, valid_token_mask=None):
        if not self._eligible(x, valid_token_mask):
            return super().forward(x, valid_token_mask)

        self._ensure_fused_runtime(x)
        normalized = self.layers[0].norm1(x)
        slot_index = 0
        for layer_index, layer in enumerate(self.layers):
            attended = layer.attention(normalized, None, self.config.causal)
            x, normalized_ffn = self._fused_residual_norm(
                attended, x, layer.norm2, slot_index
            )
            slot_index ^= 1
            ffn = layer.ffn_out(
                torch.nn.functional.gelu(
                    layer.ffn_in(normalized_ffn), approximate="none"
                )
            )
            next_norm = (
                self.layers[layer_index + 1].norm1
                if layer_index + 1 < len(self.layers)
                else self.final_norm
            )
            x, normalized = self._fused_residual_norm(
                ffn, x, next_norm, slot_index
            )
            slot_index ^= 1
        return normalized
