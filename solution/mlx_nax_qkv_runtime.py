"""Fail-closed direct-head-layout MLX NAX QKV projection for row 14.

The generated Metal 4 kernel uses the pinned Apple MLX NAX helpers already
vendored under ``solution/third_party/mlx``.  It applies the measured
BM32/BN512/BK256/WM1/WN8 half-input/fp32-accumulate packed-QKV tile and stores
the 48 logical 64-wide heads directly as contiguous Q, K and V backing
storage. Apple's MIT notice is retained in ``solution/third_party/mlx-LICENSE``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from solution.mlx_nax_runtime import (
    MLX_ROOT,
    _inline_header,
    _space_safe_torch_library_path,
)


ROOT = Path(__file__).resolve().parents[1]
KERNEL_NAME = "track3_nax_qkv_direct_head_layout"
TILE = (32, 512, 256, 1, 8)

_LOCK = threading.Lock()
_EXTENSIONS: dict[int, object] = {}
_DISABLED_REASONS: dict[int, str] = {}


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"expected one QKV source site: {old!r}")
    return source.replace(old, new)


def make_mlx_nax_qkv_source(tokens: int) -> str:
    """Generate the asserted direct-head-layout QKV Metal source."""

    if tokens < 8192 or tokens % 32:
        raise ValueError("verified direct QKV lengths are >=8192 divisible by 32")
    bm, bn, bk, wm, wn = TILE
    seen: set[Path] = set()
    helpers = _inline_header(
        MLX_ROOT / "mlx/backend/metal/kernels/steel/attn/nax.h", seen
    )
    source = r"""
#include <metal_stdlib>
using namespace metal;
template <typename U> struct Limits {
  static constexpr constant U finite_min = -metal::numeric_limits<U>::max();
};
""" + helpers + rf"""
using namespace mlx::steel;
constant int BM = {bm};
constant int BN = {bn};
constant int BK = {bk};
constant int WM = {wm};
constant int WN = {wn};
constant int N = 3072;
constant int K = 1024;
constant int M = {tokens};

template <typename T>
METAL_FUNC void qkv_mma(
    thread NAXTile<float, BM / WM / 16, BN / WN / 16>& C,
    thread NAXTile<T, BM / WM / 16, 2>& A,
    thread NAXTile<T, BN / WN / 16, 2>& B) {{
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < BM / WM / 16; ++mm) {{
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < BN / WN / 16; nn += 2) {{
      STEEL_PRAGMA_UNROLL
      for (short kk = 0; kk < 2; ++kk) {{
        NAXTile<float, BM / WM / 16, BN / WN / 16>::NAXFrag_t::mma(
            C.frag_at(mm, nn), C.frag_at(mm, nn + 1), A.frag_at(mm, kk),
            metal::false_type{{}}, B.frag_at(nn, kk), B.frag_at(nn + 1, kk),
            metal::true_type{{}});
      }}
    }}
  }}
}}

[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]]
void {KERNEL_NAME}(
    const device half* X [[buffer(0)]],
    const device half* W [[buffer(1)]],
    const device half* Bias [[buffer(2)]],
    device half* Y [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {{
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  const int output_row = int(tid.y) * BM;
  const int output_col = int(tid.x) * BN;
  const short tm = SM * short(simd_gid / WN);
  const short tn = SN * short(simd_gid % WN);
  const device half* a = X + (output_row + tm) * K;
  const device half* b = W + (output_col + tn) * K;
  const int output_slot = (output_col + tn) / 64;
  const short head_col = tn % 64;
  device half* y = Y + output_slot * M * 64 + output_row * 64;

  NAXTile<float, TM, TN> Dtile;
  Dtile.clear();
  for (int k = 0; k < K; k += BK) {{
    for (int kk = 0; kk < BK; kk += SK) {{
      NAXTile<half, TM, TK> Atile;
      NAXTile<half, TN, TK> Btile;
      Atile.load(a + k + kk, K);
      Btile.load(b + k + kk, K);
      qkv_mma<half>(Dtile, Atile, Btile);
    }}
  }}

  const short2 coord = decltype(Dtile)::NAXFrag_t::get_coord();
  STEEL_PRAGMA_UNROLL
  for (short mm = 0; mm < decltype(Dtile)::kTileRows; ++mm) {{
    STEEL_PRAGMA_UNROLL
    for (short nn = 0; nn < decltype(Dtile)::kTileCols; ++nn) {{
      thread auto& frag = Dtile.frag_at(mm, nn);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < decltype(Dtile)::NAXFrag_t::kElemRows; ++i) {{
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < decltype(Dtile)::NAXFrag_t::kElemCols; ++j) {{
          const int col = output_col + tn + nn * 16 + coord.x + j;
          const short offset =
              i * decltype(Dtile)::NAXFrag_t::kElemCols + j;
          frag[offset] += float(Bias[col]);
        }}
      }}
    }}
  }}
  if (output_row + BM <= M) {{
    Dtile.store(y + tm * 64 + head_col, 64);
  }} else {{
    Dtile.store_rows(
        y + tm * 64 + head_col, 64, M - output_row - tm);
  }}
}}
"""
    if source.count(KERNEL_NAME) != 1:
        raise RuntimeError("generated QKV source lacks one kernel declaration")
    return source


def _load_extension(tokens: int):
    from torch.utils import cpp_extension

    if shutil.which("ninja") is None:
        environment_bin = Path(sys.executable).parent
        bundled_ninja = environment_bin / "ninja"
        if bundled_ninja.is_file():
            os.environ["PATH"] = (
                f"{environment_bin}{os.pathsep}{os.environ.get('PATH', '')}"
            )
    _space_safe_torch_library_path(cpp_extension)
    source = make_mlx_nax_qkv_source(tokens)
    bridge = ROOT / "solution" / "mps_metal4_qkv_head_layout.mm"
    digest = hashlib.sha256(source.encode() + bridge.read_bytes()).hexdigest()[:12]
    extension = cpp_extension.load(
        name=f"track3_mlx_nax_qkv_{digest}",
        sources=[str(bridge)],
        extra_cflags=[
            "-std=c++17",
            f"-DTRACK3_LINEAR_M={tokens}",
            f"-DTRACK3_LINEAR_BM={TILE[0]}",
            f"-DTRACK3_LINEAR_BN={TILE[1]}",
            f"-DTRACK3_LINEAR_WM={TILE[3]}",
            f"-DTRACK3_LINEAR_WN={TILE[4]}",
        ],
        extra_ldflags=["-framework", "Foundation", "-framework", "Metal"],
        verbose=False,
    )
    status = extension.compile_qkv_head_layout_source(source, KERNEL_NAME)
    if "max_threads=256" not in status:
        raise RuntimeError(f"unexpected QKV pipeline status: {status}")
    return extension


def mlx_nax_qkv_head_project(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return contiguous Q/K/V views, or ``None`` for the existing fallback."""

    if (
        x.device.type != "mps"
        or x.dtype != torch.float32
        or x.ndim != 3
        or x.shape[0] != 1
        or x.shape[1] < 8192
        or x.shape[1] % 32
        or x.shape[2] != 1024
        or weight.device.type != "mps"
        or weight.dtype != torch.float16
        or tuple(weight.shape) != (3072, 1024)
        or not weight.is_contiguous()
        or bias.device.type != "mps"
        or bias.dtype != torch.float16
        or tuple(bias.shape) != (3072,)
        or not bias.is_contiguous()
    ):
        return None
    tokens = x.shape[1]
    if tokens in _DISABLED_REASONS:
        return None
    extension = _EXTENSIONS.get(tokens)
    if extension is None:
        with _LOCK:
            extension = _EXTENSIONS.get(tokens)
            if extension is None and tokens not in _DISABLED_REASONS:
                try:
                    extension = _load_extension(tokens)
                    _EXTENSIONS[tokens] = extension
                except Exception as exc:
                    _DISABLED_REASONS[tokens] = f"{type(exc).__name__}: {exc}"
                    return None
    try:
        flat = x.half().reshape(tokens, 1024).contiguous()
        pad_rows = (-tokens) % TILE[0]
        padded = F.pad(flat, (0, 0, 0, pad_rows)) if pad_rows else flat
        packed = extension.run_qkv_head_layout(padded, weight, bias)
        return (
            packed[:16].view(1, 16, tokens, 64),
            packed[16:32].view(1, 16, tokens, 64),
            packed[32:].view(1, 16, tokens, 64),
        )
    except Exception as exc:
        _DISABLED_REASONS[tokens] = f"{type(exc).__name__}: {exc}"
        _EXTENSIONS.pop(tokens, None)
        return None


def runtime_status() -> dict[str, object]:
    return {
        "loaded_lengths": sorted(_EXTENSIONS),
        "disabled_reasons": dict(_DISABLED_REASONS),
        "upstream_commit": "3f0bd54ff0c0af5b88530191d5df31010ce54fcd",
        "kernel_name": KERNEL_NAME,
        "tile": TILE,
    }
