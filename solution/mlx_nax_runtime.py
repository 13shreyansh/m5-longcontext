"""Lazy PyTorch bridge for the pinned Apple MLX NAX attention kernel.

The Metal source under ``solution/third_party/mlx`` is copied verbatim from
Apple MLX commit ``3f0bd54ff0c0af5b88530191d5df31010ce54fcd``.  This module
embeds the eight headers needed by the NAX attention template, instantiates the
locally verified half/BQ256/BK48/BD64/WM16/WN1 stress configuration, pads only
Q to a complete BQ256 tile, fixes function constants to the organizer's causal
route, keeps the four BD64 Q fragments resident and applies the log-base-
adjusted scale once to Q instead of every score tile, and compiles it with Metal
language 4.0. Apple's MIT notice is retained in ``solution/third_party/mlx-LICENSE``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MLX_ROOT = ROOT / "solution" / "third_party" / "mlx"
KERNEL_NAME = "steel_nax_half_bq256_bk48_bd64_wm16_wn1_alignq_qprescale"

_LOCK = threading.Lock()
_EXTENSION = None
_DISABLED_REASON: Optional[str] = None


def _inline_header(path: Path, seen: set[Path]) -> str:
    path = path.resolve()
    if path in seen:
        return ""
    seen.add(path)
    output: list[str] = []
    for line in path.read_text().splitlines():
        include = re.match(r'\s*#include\s+"([^"]+)"', line)
        if include:
            dependency = (MLX_ROOT / include.group(1)).resolve()
            if not dependency.exists():
                raise FileNotFoundError(dependency)
            output.append(_inline_header(dependency, seen))
        elif line.strip() != "#pragma once":
            output.append(line)
    return "\n".join(output) + "\n"


def _apply_bk48_safe_score_tail(source: str) -> str:
    """Round BK48 score storage while reducing only its three live columns."""

    row_reduce = """  template <typename Op>
  METAL_FUNC void row_reduce(
      thread metal::vec<T, kRowsPerThread>& vals) const thread {"""
    active_methods = """  template <typename Op, short ActiveCols>
  METAL_FUNC void row_reduce_cols(
      thread metal::vec<T, kRowsPerThread>& vals) const thread {
    static_assert(ActiveCols <= kTileCols, "active columns exceed score tile");
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < ActiveCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op, short ActiveCols>
  METAL_FUNC void row_bin_op_cols(
      thread metal::vec<T, kRowsPerThread>& vals) thread {
    static_assert(ActiveCols <= kTileCols, "active columns exceed score tile");
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < ActiveCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

"""
    if source.count(row_reduce) != 1:
        raise RuntimeError("expected one NAXTile row-reduction definition")
    source = source.replace(row_reduce, active_methods + row_reduce)

    stile_type = "using stile_t = NAXTile<AccumType, TQ, TK>;"
    if source.count(stile_type) != 2:
        raise RuntimeError("expected normal and head-split score-tile sites")
    source = source.replace(
        stile_type,
        "constexpr short TKS = TK + (TK & 1);\n"
        "    using stile_t = NAXTile<AccumType, TQ, TKS>;",
        1,
    )
    replacements = (
        (
            "Stile.template row_reduce<MaxOp>(new_max);",
            "Stile.template row_reduce_cols<MaxOp, TK>(new_max);",
        ),
        (
            "Stile.template row_bin_op<ExpSubOp>(new_max);",
            "Stile.template row_bin_op_cols<ExpSubOp, TK>(new_max);",
        ),
        (
            "Stile.template row_reduce<SumOp>(sum_score);",
            "Stile.template row_reduce_cols<SumOp, TK>(sum_score);",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 2:
            raise RuntimeError(f"expected normal and head-split operation: {old}")
        source = source.replace(old, new, 1)
    return source


def make_mlx_nax_source(
    *,
    q_prescale: bool = True,
    tile_bq: Optional[int] = None,
    tile_bk: Optional[int] = None,
    safe_bk48_tail: bool = True,
) -> str:
    """Return Metal 4 source; historical controls may request BQ128/BK32."""

    if tile_bq is None:
        tile_bq = 256 if q_prescale else 128
    if tile_bq not in (128, 256):
        raise ValueError("verified NAX source generation supports BQ128/BQ256")
    if tile_bk is None:
        tile_bk = 48
    if tile_bk not in (32, 48):
        raise ValueError("verified NAX source generation supports BK32/BK48")
    tile_warps = tile_bq // 16

    seen: set[Path] = set()
    entry = (
        MLX_ROOT
        / "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h"
    )
    body = _inline_header(entry, seen)
    if len(seen) != 8:
        raise RuntimeError(f"expected eight pinned MLX headers, embedded {len(seen)}")
    replacements = {
        "constant bool align_Q [[function_constant(200)]];":
            "constant bool align_Q = true;",
        "constant bool align_K [[function_constant(201)]];":
            f"constant bool align_K = {'true' if tile_bk == 32 else 'false'};",
        "constant bool has_mask [[function_constant(300)]];":
            "constant bool has_mask = false;",
        "constant bool do_causal [[function_constant(301)]];":
            "constant bool do_causal = true;",
        "constant bool has_sinks [[function_constant(302)]];":
            "constant bool has_sinks = false;",
    }
    for old, new in replacements.items():
        if body.count(old) != 1:
            raise RuntimeError(f"expected exactly one upstream declaration: {old}")
        body = body.replace(old, new)

    if not q_prescale:
        prefix = r"""
#include <metal_stdlib>
using namespace metal;
template <typename U> struct Limits {
  static constexpr constant U finite_min = -metal::numeric_limits<U>::max();
};
#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) \
  func<__VA_ARGS__>;
"""
        instantiation = rf"""
instantiate_kernel(
    "{KERNEL_NAME}",
    attention_nax,
    half, {tile_bq}, {tile_bk}, 64, {tile_warps}, 1, half, float)
"""
        source = prefix + body + instantiation
        return (
            _apply_bk48_safe_score_tail(source)
            if tile_bk == 48 and safe_bk48_tail
            else source
        )

    def replace_once(old: str, new: str) -> None:
        nonlocal body
        if body.count(old) != 1:
            raise RuntimeError(f"expected exactly one NAX transform site: {old!r}")
        body = body.replace(old, new)

    replace_once(
        "  const short lim_rows_k = params->kL_rem;\n\n"
        "  // Loop over KV seq length",
        "  const short lim_rows_k = params->kL_rem;\n\n"
        "  // BD64 has four Q fragments. Scale and retain them once rather\n"
        "  // than reloading Q and scaling every score tile in the KV loop.\n"
        "  NAXTile<T, 1, 1> Qtiles[TD];\n"
        "  STEEL_PRAGMA_UNROLL\n"
        "  for (short id = 0; id < TD; id++) {\n"
        "    Qtiles[id].load(\n"
        "        Q + id * kU, int(params->Q_strides[2]));\n"
        "    STEEL_PRAGMA_UNROLL\n"
        "    for (short ii = 0; ii < Qtiles[id].kElemsPerTile; ii++) {\n"
        "      Qtiles[id].elems()[ii] *= T(scale2);\n"
        "    }\n"
        "  }\n\n"
        "  // Loop over KV seq length",
    )
    replace_once(
        "          NAXTile<T, 1, 1> Qtile;\n"
        "          NAXTile<T, 2, 1> Ktile;",
        "          NAXTile<T, 2, 1> Ktile;",
    )
    replace_once(
        "          const int Q_load_off = iq * kU * int(params->Q_strides[2]) + id * kU;\n"
        "          const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;",
        "          const int K_load_off = ik * kU * int(params->K_strides[2]) + id * kU;",
    )
    replace_once(
        "          if (!align_Q && is_last_q) {\n"
        "            Qtile.load_rows(\n"
        "                Q + Q_load_off,\n"
        "                int(params->Q_strides[2]),\n"
        "                lim_rows_q - iq * kU);\n"
        "          } else {\n"
        "            Qtile.load(Q + Q_load_off, int(params->Q_strides[2]));\n"
        "          }\n\n",
        "",
    )
    replace_once(
        "              Qtile.frag_at(0, 0),",
        "              Qtiles[id].frag_at(0, 0),",
    )
    score_scale = (
        "    // Scale S\n"
        "    STEEL_PRAGMA_UNROLL\n"
        "    for (short ii = 0; ii < stile_t::kElemsPerTile; ii++) {\n"
        "      Stile.elems()[ii] *= float(scale2);\n"
        "    }\n\n"
    )
    if body.count(score_scale) != 2:
        raise RuntimeError("expected ordinary and head-split NAX scale sites")
    body = body.replace(score_scale, "", 1)

    prefix = r"""
#include <metal_stdlib>
using namespace metal;
template <typename U> struct Limits {
  static constexpr constant U finite_min = -metal::numeric_limits<U>::max();
};
#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) \
  func<__VA_ARGS__>;
"""
    instantiation = rf"""
instantiate_kernel(
    "{KERNEL_NAME}",
    attention_nax,
    half, {tile_bq}, {tile_bk}, 64, {tile_warps}, 1, half, float)
"""
    source = prefix + body + instantiation
    return (
        _apply_bk48_safe_score_tail(source)
        if tile_bk == 48 and safe_bk48_tail
        else source
    )


def _space_safe_torch_library_path(cpp_extension) -> None:
    """Work around PyTorch 2.8's unquoted ``-L`` path in its Ninja file."""

    torch_library_path = Path(cpp_extension.TORCH_LIB_PATH).resolve()
    if " " not in str(torch_library_path):
        return
    digest = hashlib.sha256(str(torch_library_path).encode()).hexdigest()[:12]
    link = Path(os.getenv("TMPDIR", "/tmp")) / f"track3-torch-lib-{digest}"
    try:
        if link.is_symlink() and link.resolve() != torch_library_path:
            link.unlink()
        if not link.exists():
            link.symlink_to(torch_library_path, target_is_directory=True)
    except OSError as exc:
        raise RuntimeError("could not create space-safe temporary Torch lib link") from exc
    cpp_extension.TORCH_LIB_PATH = str(link)


def _load_extension():
    from torch.utils import cpp_extension

    if shutil.which("ninja") is None:
        environment_bin = Path(sys.executable).parent
        bundled_ninja = environment_bin / "ninja"
        if bundled_ninja.is_file():
            os.environ["PATH"] = f"{environment_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    _space_safe_torch_library_path(cpp_extension)
    source = make_mlx_nax_source()
    bridge = ROOT / "solution" / "mps_metal4_attention.mm"
    digest = hashlib.sha256(source.encode() + bridge.read_bytes()).hexdigest()[:12]
    extension = cpp_extension.load(
        name=f"track3_mlx_nax_{digest}",
        sources=[str(bridge)],
        extra_cflags=["-std=c++17", "-DTRACK3_NAX_BK=48"],
        extra_ldflags=["-framework", "Foundation", "-framework", "Metal"],
        verbose=False,
    )
    extension.compile_metal4_source(source, KERNEL_NAME)
    return extension


def mlx_nax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> Optional[torch.Tensor]:
    """Run the exact NAX specialization, or return ``None`` for safe fallback."""

    global _EXTENSION, _DISABLED_REASON
    if (
        q.device.type != "mps"
        or q.dtype != torch.float16
        or k.dtype != torch.float16
        or v.dtype != torch.float16
        or q.ndim != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[-1] != 64
        or q.shape[-2] % 32 != 0
        or not q.is_contiguous()
        or not k.is_contiguous()
        or not v.is_contiguous()
    ):
        return None
    if _DISABLED_REASON is not None:
        return None
    if _EXTENSION is None:
        with _LOCK:
            if _EXTENSION is None and _DISABLED_REASON is None:
                try:
                    _EXTENSION = _load_extension()
                except Exception as exc:
                    _DISABLED_REASON = f"{type(exc).__name__}: {exc}"
                    return None
    try:
        length = q.shape[-2]
        pad_rows = (-length) % 256
        padded_q = F.pad(q, (0, 0, 0, pad_rows)) if pad_rows else q
        output = _EXTENSION.run_nax_attention(padded_q, k, v, float(scale))
        return output[..., :length, :]
    except Exception as exc:
        _DISABLED_REASON = f"{type(exc).__name__}: {exc}"
        return None


def runtime_status() -> dict[str, object]:
    """Expose truthful diagnostic state without changing dispatch behavior."""

    return {
        "loaded": _EXTENSION is not None,
        "disabled_reason": _DISABLED_REASON,
        "upstream_commit": "3f0bd54ff0c0af5b88530191d5df31010ce54fcd",
        "kernel_name": KERNEL_NAME,
        "safe_bk48_score_tail": True,
    }
