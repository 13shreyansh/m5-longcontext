#!/usr/bin/env python3
"""Verify committed generated kernels and licences against pinned checkouts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "artifacts" / "upstreams" / "triton-msl"
PINNED_COMMIT = "182c1820fd24a836d565e1da842f28414de64084"
MLX_UPSTREAM = ROOT / "artifacts" / "upstreams" / "mlx"
MLX_PINNED_COMMIT = "3f0bd54ff0c0af5b88530191d5df31010ce54fcd"
sys.path.insert(0, str(ROOT))

from solution.case14_metal import (  # noqa: E402
    CAUSAL_FLASH_ATTENTION_FP32_HD64,
)
from solution.case14_metal_fp16 import (  # noqa: E402
    CAUSAL_FLASH_ATTENTION_FP16_HD64,
)
from solution.case14_metal_fp16_fast_exp import (  # noqa: E402
    CAUSAL_FLASH_ATTENTION_FP16_FAST_EXP_HD64,
    make_fast_exp_source,
)
from solution.mlx_nax_runtime import make_mlx_nax_source  # noqa: E402
from solution.mlx_nax_qkv_runtime import (  # noqa: E402
    make_mlx_nax_qkv_source,
)
from solution.metal_kernels import (  # noqa: E402
    FUSED_RESIDUAL_NORM_FP32_32,
    FUSED_RESIDUAL_NORM_FP32_64,
    FUSED_RESIDUAL_NORM_FP32_128,
)
def verified_commit(path: Path, expected: str) -> str:
    if not path.is_dir():
        raise RuntimeError(
            f"pinned checkout is unavailable: {path}\n"
            "run ./scripts/acquire_solution_upstreams.sh first"
        )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected:
        raise AssertionError(f"upstream commit={commit}, expected={expected}")
    return commit


def main() -> int:
    commit = verified_commit(UPSTREAM, PINNED_COMMIT)
    mlx_commit = verified_commit(MLX_UPSTREAM, MLX_PINNED_COMMIT)

    sys.path.insert(0, str(UPSTREAM))
    from triton_msl.codegen._msl_templates import (
        make_flash_attention_kernel_simdgroup,
        make_fused_residual_norm_kernel,
    )

    def generated(dtype: str, kernel_name: str) -> str:
        return make_flash_attention_kernel_simdgroup(
            head_dim=64,
            BLOCK_M=32,
            BLOCK_N=64,
            causal=True,
            out_dtype=dtype,
            kernel_name=kernel_name,
        )

    expected_fp32 = generated("fp32", "causal_flash_attention_fp32_hd64")
    expected_fp16 = generated("fp16", "causal_flash_attention_fp16_hd64")
    if CAUSAL_FLASH_ATTENTION_FP32_HD64.strip() != expected_fp32.strip():
        raise AssertionError("committed fp32 Metal source differs from generator")
    if CAUSAL_FLASH_ATTENTION_FP16_HD64.strip() != expected_fp16.strip():
        raise AssertionError("committed fp16 Metal source differs from generator")
    expected_fast_exp = make_fast_exp_source(expected_fp16)
    if (
        CAUSAL_FLASH_ATTENTION_FP16_FAST_EXP_HD64.strip()
        != expected_fast_exp.strip()
    ):
        raise AssertionError("fast-exp Metal source differs from asserted transform")
    expected_fused128 = make_fused_residual_norm_kernel(
        block_size=128, dtype="fp32", eps=1e-5
    )
    if FUSED_RESIDUAL_NORM_FP32_128.strip() != expected_fused128.strip():
        raise AssertionError("committed fused-128 source differs from generator")
    expected_fused64 = make_fused_residual_norm_kernel(
        block_size=64, dtype="fp32", eps=1e-5
    )
    if FUSED_RESIDUAL_NORM_FP32_64.strip() != expected_fused64.strip():
        raise AssertionError("committed fused-64 source differs from generator")
    expected_fused32 = make_fused_residual_norm_kernel(
        block_size=32, dtype="fp32", eps=1e-5
    )
    if FUSED_RESIDUAL_NORM_FP32_32.strip() != expected_fused32.strip():
        raise AssertionError("committed fused-32 source differs from generator")

    upstream_licence = (UPSTREAM / "LICENSE").read_bytes()
    retained_licence = (
        ROOT / "solution" / "third_party" / "triton-msl-LICENSE"
    ).read_bytes()
    if retained_licence != upstream_licence:
        raise AssertionError("retained triton-msl licence differs from upstream")

    mlx_files = (
        "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h",
        "mlx/backend/metal/kernels/steel/attn/nax.h",
        "mlx/backend/metal/kernels/steel/attn/params.h",
        "mlx/backend/metal/kernels/steel/attn/transforms.h",
        "mlx/backend/metal/kernels/steel/defines.h",
        "mlx/backend/metal/kernels/steel/utils.h",
        "mlx/backend/metal/kernels/steel/utils/integral_constant.h",
        "mlx/backend/metal/kernels/steel/utils/type_traits.h",
    )
    vendored_mlx_root = ROOT / "solution" / "third_party" / "mlx"
    for relative in mlx_files:
        if (vendored_mlx_root / relative).read_bytes() != (
            MLX_UPSTREAM / relative
        ).read_bytes():
            raise AssertionError(f"vendored MLX source differs: {relative}")
    if (ROOT / "solution" / "third_party" / "mlx-LICENSE").read_bytes() != (
        MLX_UPSTREAM / "LICENSE"
    ).read_bytes():
        raise AssertionError("retained MLX licence differs from upstream")
    metal4_source = make_mlx_nax_source()
    if metal4_source.count("attention_nax") < 2:
        raise AssertionError("generated MLX NAX source is incomplete")
    if "half, 256, 48, 64, 16, 1, half, float" not in metal4_source:
        raise AssertionError("generated MLX NAX source lacks verified BQ256/BK48 tile")
    if "constant bool align_Q = true;" not in metal4_source:
        raise AssertionError("generated MLX NAX source lacks aligned-Q specialization")
    if "constant bool align_K = false;" not in metal4_source:
        raise AssertionError("generated MLX NAX source lacks partial-key handling")
    if "NAXTile<AccumType, TQ, TKS>" not in metal4_source:
        raise AssertionError("generated MLX NAX source lacks safe BK48 score storage")
    if metal4_source.count("row_reduce_cols<MaxOp, TK>") != 1:
        raise AssertionError("generated MLX NAX source lacks active-column max")
    if metal4_source.count("row_bin_op_cols<ExpSubOp, TK>") != 1:
        raise AssertionError("generated MLX NAX source lacks active-column exp")
    if metal4_source.count("row_reduce_cols<SumOp, TK>") != 1:
        raise AssertionError("generated MLX NAX source lacks active-column sum")
    if metal4_source.count("Qtiles[id].elems()[ii] *= T(scale2);") != 1:
        raise AssertionError("generated MLX NAX source lacks one-time Q prescale")
    if metal4_source.count("Stile.elems()[ii] *= float(scale2);") != 1:
        raise AssertionError("ordinary NAX score-scale loop was not removed exactly")
    bq128_control = make_mlx_nax_source(tile_bq=128, tile_bk=32)
    if "half, 128, 32, 64, 8, 1, half, float" not in bq128_control:
        raise AssertionError("generated MLX NAX source lacks historical BQ128 control")
    pre_prescale_control = make_mlx_nax_source(q_prescale=False, tile_bk=32)
    if pre_prescale_control.count(
        "Stile.elems()[ii] *= float(scale2);"
    ) != 2:
        raise AssertionError("historical pre-prescale source was not preserved")
    qkv_source = make_mlx_nax_qkv_source(100000)
    if "constant int BM = 32;" not in qkv_source:
        raise AssertionError("generated MLX QKV source lacks verified BM32 tile")
    if "constant int BN = 512;" not in qkv_source:
        raise AssertionError("generated MLX QKV source lacks verified BN512 tile")
    if "constant int WM = 1;" not in qkv_source or "constant int WN = 8;" not in qkv_source:
        raise AssertionError("generated MLX QKV source lacks verified warp geometry")
    if "output_slot * M * 64" not in qkv_source:
        raise AssertionError("generated MLX QKV source lacks direct head-major store")

    print(f"triton-msl commit: OK ({commit})")
    print("generated float32 Metal source: OK")
    print("generated float16 Metal source: OK")
    print("asserted fast-exp Metal transform: OK")
    print("generated fused-128 Metal source: OK")
    print("generated fused-64 Metal source: OK")
    print("generated fused-32 Metal source: OK")
    print("retained triton-msl MIT licence: OK")
    print(f"MLX commit: OK ({mlx_commit})")
    print("vendored MLX NAX source: OK (8/8 files)")
    print("asserted MLX BQ256 transform and historical controls: OK")
    print("asserted MLX BM32/BN512 direct-head QKV transform: OK")
    print("retained MLX MIT licence: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
