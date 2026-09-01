from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from solution.case14_metal import CAUSAL_FLASH_ATTENTION_FP32_HD64
from solution.case14_metal_fp16 import CAUSAL_FLASH_ATTENTION_FP16_HD64
from solution.case14_metal_fp16_fast_exp import (
    CAUSAL_FLASH_ATTENTION_FP16_FAST_EXP_HD64,
    FAST_EXP_REPLACEMENTS,
)
from solution.optimized_transformer import (
    OptimizedSelfAttention,
    UserOptimizedTransformer,
    _specialize_case14_source,
)
from solution.mlx_nax_runtime import KERNEL_NAME, make_mlx_nax_source
from solution.mlx_nax_qkv_runtime import make_mlx_nax_qkv_source
import solution.mlx_nax_runtime as nax_runtime
import solution.mlx_nax_qkv_runtime as qkv_runtime
import solution.optimized_transformer as optimized_module


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_PATH = ROOT / "official" / "torch_transformer_benchmark.py"
if not OFFICIAL_PATH.is_file():
    pytest.skip(
        "authorized organizer benchmark is absent; see official/README.md",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location("official_for_tests", OFFICIAL_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not import {OFFICIAL_PATH}")
official = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = official
SPEC.loader.exec_module(official)


def make_models(device="cpu", dtype=torch.float32):
    config = official.TransformerConfig(
        batch_size=2,
        seq_len=16,
        d_model=32,
        num_heads=4,
        ffn_dim=32,
        num_layers=2,
        causal=True,
    )
    torch.manual_seed(1234)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    return (
        config,
        baseline.to(device=device, dtype=dtype).eval(),
        optimized.to(device=device, dtype=dtype).eval(),
    )


def assert_organizer_close(baseline, optimized, x, mask):
    with torch.inference_mode():
        expected = baseline(x, mask)
        actual = optimized(x, mask)
        if x.device.type == "mps":
            torch.mps.synchronize()
    result = official.compare_outputs(expected, actual, rtol=0.02, atol=0.002)
    assert result.passed, result


def test_state_dict_keys_match_reference_exactly():
    _, baseline, optimized = make_models()
    assert list(baseline.state_dict()) == list(optimized.state_dict())


def test_reloading_weights_refreshes_packed_qkv():
    config, _, optimized = make_models()
    torch.manual_seed(9876)
    second_baseline = official.BaselineTransformer(config).eval()
    official.copy_model_weights(second_baseline, optimized, strict=True)
    x, mask = official.generate_random_case(
        config, torch.device("cpu"), torch.float32, 111, 0.0, 1.0
    )
    assert_organizer_close(second_baseline, optimized, x, mask)


def test_static_graph_gate_rejects_cpu():
    config = official.TransformerConfig(
        batch_size=4,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    candidate = UserOptimizedTransformer(config).eval()
    x = torch.zeros((4, 128, 128), dtype=torch.float32)
    mask = torch.ones((4, 128), dtype=torch.bool)
    with torch.inference_mode():
        assert not candidate._static_graph_eligible(x, mask)


def test_static_graph_modes_match_only_promoted_configurations():
    expected = (
        (1, 128, 4, "freeze"),
        (4, 128, 4, "optimize"),
        (64, 128, 1, "freeze"),
        (64, 32, 4, "freeze"),
    )
    for batch_size, seq_len, num_heads, mode in expected:
        config = official.TransformerConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=128,
            num_heads=num_heads,
            ffn_dim=128,
            num_layers=4,
            causal=True,
        )
        assert UserOptimizedTransformer(config)._static_graph_mode == mode

    near_miss = official.TransformerConfig(
        batch_size=64,
        seq_len=127,
        d_model=128,
        num_heads=1,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    assert UserOptimizedTransformer(near_miss)._static_graph_mode is None
    rejected_margin = official.TransformerConfig(
        batch_size=64,
        seq_len=128,
        d_model=128,
        num_heads=16,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    assert UserOptimizedTransformer(rejected_margin)._static_graph_mode is None


def test_mps_case2_frozen_graph_is_bit_identical_to_eager_solution():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=1,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(8247)
    baseline = official.BaselineTransformer(config)
    eager = UserOptimizedTransformer(config)
    candidate = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, eager, strict=True)
    official.copy_model_weights(baseline, candidate, strict=True)
    baseline = baseline.to("mps").eval()
    eager = eager.to("mps").eval()
    candidate = candidate.to("mps").eval()
    eager._static_graph_disabled_reason = "test eager control"
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 8248, 0.0, 1.0
    )
    with torch.inference_mode():
        expected = baseline(x, mask)
        eager_output = eager(x, mask)
        actual = candidate(x, mask)
        torch.mps.synchronize()
    comparison = official.compare_outputs(
        expected, actual, rtol=0.02, atol=0.002
    )
    assert comparison.passed, comparison
    assert torch.equal(eager_output, actual)
    assert candidate._static_graph_mode == "freeze"
    assert candidate.__dict__["_static_all_valid_graph"] is not None


def test_mps_case3_graph_is_correct_unregistered_and_invalidated():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=4,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(8096)
    baseline = official.BaselineTransformer(config)
    candidate = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, candidate, strict=True)
    baseline = baseline.to("mps").eval()
    candidate = candidate.to("mps").eval()
    keys_before = list(candidate.state_dict())
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 8097, 0.0, 1.0
    )
    with torch.inference_mode():
        assert candidate._static_graph_eligible(x, mask)
        near_x = torch.zeros((4, 127, 128), device="mps")
        assert not candidate._static_graph_eligible(near_x, mask[:, :127])
    with torch.enable_grad():
        assert not candidate._static_graph_eligible(x, mask)
    candidate.train()
    with torch.inference_mode():
        assert not candidate._static_graph_eligible(x, mask)
    candidate.eval()
    assert_organizer_close(baseline, candidate, x, mask)
    graph = candidate.__dict__["_static_all_valid_graph"]
    assert graph is not None
    assert candidate._static_graph_disabled_reason is None
    assert list(candidate.state_dict()) == keys_before
    assert all(module is not graph for module in candidate.modules())

    padded_x, padded_mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 8098, 0.25, 1.0
    )
    assert_organizer_close(baseline, candidate, padded_x, padded_mask)
    assert candidate.__dict__["_static_all_valid_graph"] is graph

    torch.manual_seed(8099)
    second_baseline = official.BaselineTransformer(config).to("mps").eval()
    official.copy_model_weights(second_baseline, candidate, strict=True)
    assert candidate.__dict__["_static_all_valid_graph"] is None
    x2, mask2 = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 8100, 0.0, 1.0
    )
    assert_organizer_close(second_baseline, candidate, x2, mask2)
    assert candidate.__dict__["_static_all_valid_graph"] is not None

    candidate.train()
    assert candidate.__dict__["_static_all_valid_graph"] is None
    candidate.eval()
    candidate._apply(lambda tensor: tensor)
    assert candidate.__dict__["_static_all_valid_graph"] is None


def test_mps_case3_trace_failure_falls_back(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=4,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(8101)
    baseline = official.BaselineTransformer(config)
    candidate = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, candidate, strict=True)
    baseline = baseline.to("mps").eval()
    candidate = candidate.to("mps").eval()

    def reject_trace(*_args, **_kwargs):
        raise RuntimeError("synthetic trace failure")

    monkeypatch.setattr(torch.jit, "trace", reject_trace)
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 8102, 0.0, 1.0
    )
    assert_organizer_close(baseline, candidate, x, mask)
    assert candidate.__dict__["_static_all_valid_graph"] is None
    assert candidate._static_graph_disabled_reason == (
        "RuntimeError: synthetic trace failure"
    )


def test_reloading_weights_invalidates_half_qkv_cache():
    config, _, optimized = make_models()
    attention = optimized.layers[0].attention
    x = torch.randn((2, 16, 32), dtype=torch.float32)
    first_weight, first_bias = attention._half_qkv_parameters(x)
    assert first_weight.dtype == torch.float16
    assert first_bias.dtype == torch.float16

    torch.manual_seed(8765)
    second_baseline = official.BaselineTransformer(config).eval()
    official.copy_model_weights(second_baseline, optimized, strict=True)
    assert attention._qkv_weight_half is None
    assert attention._qkv_bias_half is None
    second_weight, second_bias = attention._half_qkv_parameters(x)
    assert second_weight.dtype == torch.float16
    assert second_bias.dtype == torch.float16
    assert second_weight.data_ptr() != first_weight.data_ptr()


def test_reloading_weights_invalidates_half_dense_caches():
    config, _, optimized = make_models()
    layer = optimized.layers[0]
    attention = layer.attention
    x = torch.randn((2, 16, 32), dtype=torch.float32)
    out_weight, out_bias = attention._half_out_proj_parameters(x)
    ffn_parameters = layer._half_ffn_parameters(x)
    assert out_weight.dtype == torch.float16
    assert out_bias.dtype == torch.float16
    assert all(parameter.dtype == torch.float16 for parameter in ffn_parameters)

    torch.manual_seed(7654)
    second_baseline = official.BaselineTransformer(config).eval()
    official.copy_model_weights(second_baseline, optimized, strict=True)
    assert attention._out_proj_weight_half is None
    assert attention._out_proj_bias_half is None
    assert layer._ffn_in_weight_half is None
    assert layer._ffn_in_bias_half is None
    assert layer._ffn_out_weight_half is None
    assert layer._ffn_out_bias_half is None

    second_out_weight, _ = attention._half_out_proj_parameters(x)
    second_ffn_parameters = layer._half_ffn_parameters(x)
    assert second_out_weight.data_ptr() != out_weight.data_ptr()
    assert second_ffn_parameters[0].data_ptr() != ffn_parameters[0].data_ptr()


def test_mask_cache_changes_route_for_new_mask():
    config, baseline, optimized = make_models()
    x, all_valid = official.generate_random_case(
        config, torch.device("cpu"), torch.float32, 222, 0.0, 1.0
    )
    assert_organizer_close(baseline, optimized, x, all_valid)
    padded_x, padded_mask = official.generate_random_case(
        config, torch.device("cpu"), torch.float32, 223, 0.5, 1.0
    )
    assert not bool(padded_mask.all())
    assert_organizer_close(baseline, optimized, padded_x, padded_mask)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mps_declared_dtypes(dtype):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config, baseline, optimized = make_models("mps", dtype)
    for padding_ratio in (0.0, 0.25):
        x, mask = official.generate_random_case(
            config, torch.device("mps"), dtype, 333, padding_ratio, 1.0
        )
        assert_organizer_close(baseline, optimized, x, mask)


def test_mps_width_1024_fused_norm_route():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=2,
        seq_len=8,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    torch.manual_seed(1234)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to("mps").eval()
    optimized = optimized.to("mps").eval()
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 444, 0.0, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    assert optimized._fused_norm_library is not None

    padded_x, padded_mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 445, 0.25, 1.0
    )
    assert_organizer_close(baseline, optimized, padded_x, padded_mask)


def test_case14_metal_sources_compile_on_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    fp32_library = torch.mps.compile_shader(CAUSAL_FLASH_ATTENTION_FP32_HD64)
    fp16_library = torch.mps.compile_shader(CAUSAL_FLASH_ATTENTION_FP16_HD64)
    fast_exp_library = torch.mps.compile_shader(
        CAUSAL_FLASH_ATTENTION_FP16_FAST_EXP_HD64
    )
    assert hasattr(fp32_library, "causal_flash_attention_fp32_hd64")
    assert hasattr(fp16_library, "causal_flash_attention_fp16_hd64")
    assert hasattr(fast_exp_library, "causal_flash_attention_fp16_hd64")


def test_fast_exp_source_changes_only_asserted_sites():
    restored = CAUSAL_FLASH_ATTENTION_FP16_FAST_EXP_HD64
    for old, new in FAST_EXP_REPLACEMENTS:
        assert restored.count(new) == 2
        restored = restored.replace(new, old)
    assert restored == CAUSAL_FLASH_ATTENTION_FP16_HD64


def test_mlx_nax_source_has_asserted_specialization_and_licence():
    source = make_mlx_nax_source()
    assert f'"{KERNEL_NAME}"' in source
    assert "attention_nax" in source
    assert "half, 256, 48, 64, 16, 1, half, float" in source
    assert "constant bool align_K = false;" in source
    assert "NAXTile<AccumType, TQ, TKS>" in source
    assert source.count("row_reduce_cols<MaxOp, TK>") == 1
    assert source.count("row_bin_op_cols<ExpSubOp, TK>") == 1
    assert source.count("row_reduce_cols<SumOp, TK>") == 1
    unsafe_bk48 = make_mlx_nax_source(tile_bk=48, safe_bk48_tail=False)
    assert "NAXTile<AccumType, TQ, TKS>" not in unsafe_bk48
    assert "half, 128, 32, 64, 8, 1, half, float" in make_mlx_nax_source(
        tile_bq=128, tile_bk=32
    )
    assert "constant bool align_Q = true;" in source
    assert "constant bool align_K = true;" in make_mlx_nax_source(tile_bk=32)
    assert "constant bool do_causal = true;" in source
    assert "function_constant(301)" not in source
    assert source.count("Qtiles[id].elems()[ii] *= T(scale2);") == 1
    assert source.count("Stile.elems()[ii] *= float(scale2);") == 1
    assert "NAXTile<T, 1, 1> Qtile;" not in source
    assert (ROOT / "solution" / "third_party" / "mlx-LICENSE").is_file()


def test_mlx_nax_runtime_launch_failure_disables_route(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    class BrokenExtension:
        @staticmethod
        def run_nax_attention(*_args):
            raise RuntimeError("synthetic launch failure")

    monkeypatch.setattr(nax_runtime, "_EXTENSION", BrokenExtension())
    monkeypatch.setattr(nax_runtime, "_DISABLED_REASON", None)
    q = torch.zeros((1, 1, 32, 64), device="mps", dtype=torch.float16)
    assert nax_runtime.mlx_nax_attention(q, q, q, 0.125) is None
    assert "synthetic launch failure" in nax_runtime.runtime_status()[
        "disabled_reason"
    ]


def test_mlx_nax_qkv_source_has_promoted_exact_geometry():
    source = make_mlx_nax_qkv_source(100000)
    assert "constant int BM = 32;" in source
    assert "constant int BN = 512;" in source
    assert "constant int BK = 256;" in source
    assert "constant int WM = 1;" in source
    assert "constant int WN = 8;" in source
    assert "output_slot * M * 64" in source
    assert "Dtile.store_rows" in source
    assert qkv_runtime.runtime_status()["tile"] == (32, 512, 256, 1, 8)


def test_mlx_nax_qkv_runtime_rejects_unsupported_input_before_load():
    sentinel = object()
    original_extensions = qkv_runtime._EXTENSIONS
    original_disabled = qkv_runtime._DISABLED_REASONS
    try:
        qkv_runtime._EXTENSIONS = {8192: sentinel}
        qkv_runtime._DISABLED_REASONS = {}
        x = torch.empty((1, 32, 1024), dtype=torch.float32)
        weight = torch.empty((3072, 1024), dtype=torch.float16)
        bias = torch.empty((3072,), dtype=torch.float16)
        assert qkv_runtime.mlx_nax_qkv_head_project(x, weight, bias) is None
        assert qkv_runtime._EXTENSIONS == {8192: sentinel}
        assert qkv_runtime._DISABLED_REASONS == {}
    finally:
        qkv_runtime._EXTENSIONS = original_extensions
        qkv_runtime._DISABLED_REASONS = original_disabled


def test_mlx_nax_qkv_runtime_launch_failure_disables_length(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    class BrokenExtension:
        @staticmethod
        def run_qkv_head_layout(*_args):
            raise RuntimeError("synthetic QKV launch failure")

    monkeypatch.setattr(qkv_runtime, "_EXTENSIONS", {8192: BrokenExtension()})
    monkeypatch.setattr(qkv_runtime, "_DISABLED_REASONS", {})
    x = torch.zeros((1, 8192, 1024), device="mps", dtype=torch.float32)
    weight = torch.zeros((3072, 1024), device="mps", dtype=torch.float16)
    bias = torch.zeros((3072,), device="mps", dtype=torch.float16)
    assert qkv_runtime.mlx_nax_qkv_head_project(x, weight, bias) is None
    assert "synthetic QKV launch failure" in qkv_runtime.runtime_status()[
        "disabled_reasons"
    ][8192]


@pytest.mark.parametrize(
    "mutation",
    [
        "cpu",
        "float32",
        "rank3",
        "shape_mismatch",
        "head_dim",
        "sequence_remainder",
        "noncontiguous",
    ],
)
def test_mlx_nax_runtime_rejects_inputs_outside_exact_envelope(mutation):
    """Unsupported inputs must fail closed before loading native code."""

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    q = torch.zeros((1, 1, 128, 64), device=device, dtype=torch.float16)
    k = q
    v = q
    if mutation == "cpu":
        q = k = v = q.cpu()
    elif mutation == "float32":
        q = k = v = q.float()
    elif mutation == "rank3":
        q = k = v = q.squeeze(0)
    elif mutation == "shape_mismatch":
        k = k[:, :, :96]
    elif mutation == "head_dim":
        q = k = v = q[..., :32]
    elif mutation == "sequence_remainder":
        q = k = v = q[:, :, :127]
    elif mutation == "noncontiguous":
        storage = torch.zeros(
            (1, 1, 128, 128), device=device, dtype=torch.float16
        )
        q = k = v = storage[..., ::2]
        assert q.shape == (1, 1, 128, 64) and not q.is_contiguous()
    else:  # pragma: no cover - protects this test's own table.
        raise AssertionError(mutation)

    sentinel = object()
    original_extension = nax_runtime._EXTENSION
    original_disabled_reason = nax_runtime._DISABLED_REASON
    try:
        nax_runtime._EXTENSION = sentinel
        nax_runtime._DISABLED_REASON = None
        assert nax_runtime.mlx_nax_attention(q, k, v, 0.125) is None
        assert nax_runtime._EXTENSION is sentinel
        assert nax_runtime._DISABLED_REASON is None
    finally:
        nax_runtime._EXTENSION = original_extension
        nax_runtime._DISABLED_REASON = original_disabled_reason


def test_mlx_nax_runtime_supports_nondefault_batch_heads_and_length():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    torch.manual_seed(779)
    q = torch.randn((2, 3, 96, 64), device="mps", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attention = OptimizedSelfAttention(192, 3).to(
        device="mps", dtype=torch.float16
    )
    expected = attention._explicit_attention(q, k, v, None, True)
    actual = nax_runtime.mlx_nax_attention(q, k, v, 0.125)
    torch.mps.synchronize()
    assert actual is not None
    result = official.compare_outputs(expected, actual, rtol=0.02, atol=0.002)
    assert result.passed, result


def test_mlx_nax_runtime_pads_only_query_to_bq256(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    observed = {}

    class RecordingExtension:
        @staticmethod
        def run_nax_attention(q, k, v, _scale):
            observed["shapes"] = (q.shape, k.shape, v.shape)
            return torch.zeros_like(q)

    monkeypatch.setattr(nax_runtime, "_EXTENSION", RecordingExtension())
    monkeypatch.setattr(nax_runtime, "_DISABLED_REASON", None)
    q = torch.zeros((2, 3, 96, 64), device="mps", dtype=torch.float16)
    output = nax_runtime.mlx_nax_attention(q, q, q, 0.125)
    assert output is not None and output.shape == q.shape
    assert observed["shapes"] == (
        torch.Size((2, 3, 256, 64)),
        torch.Size((2, 3, 96, 64)),
        torch.Size((2, 3, 96, 64)),
    )


@pytest.mark.parametrize(
    ("seq_len", "dtype", "expected"),
    [
        (32767, torch.float32, False),
        (32768, torch.float32, True),
        (32769, torch.float32, True),
        (32768, torch.float16, False),
        (32768, torch.bfloat16, False),
    ],
)
def test_internal_half_route_exact_boundary(seq_len, dtype, expected):
    x = torch.empty((1, seq_len, 1024), device="meta", dtype=dtype)
    assert OptimizedSelfAttention._internal_fp16_attention_eligible(x) is expected
    assert OptimizedSelfAttention._fast_exp_attention_eligible(x) is expected
    assert OptimizedSelfAttention._half_qkv_projection_eligible(x) is expected


def test_prefix_valid_length_accepts_only_boolean_prefix_masks():
    seq_len = 128
    positions = torch.arange(seq_len)
    assert OptimizedSelfAttention._prefix_valid_length(None, seq_len) == seq_len
    assert (
        OptimizedSelfAttention._prefix_valid_length(
            (positions < 97)[None, :], seq_len
        )
        == 97
    )
    with_hole = (positions < 97)[None, :]
    with_hole[0, 20] = False
    with_hole[0, 100] = True
    assert OptimizedSelfAttention._prefix_valid_length(with_hole, seq_len) is None
    assert (
        OptimizedSelfAttention._prefix_valid_length(
            torch.ones((1, seq_len), dtype=torch.int32), seq_len
        )
        is None
    )
    assert (
        OptimizedSelfAttention._prefix_valid_length(
            torch.ones((2, seq_len), dtype=torch.bool), seq_len
        )
        is None
    )


def test_case14_constant_specialization_preserves_kernel_abi():
    values = {"H": 16, "N_CTX": 100000, "q_sk": 1}
    specialized = _specialize_case14_source(
        CAUSAL_FLASH_ATTENTION_FP16_FAST_EXP_HD64, values
    )
    for name, value in values.items():
        assert f"const uint {name} = {value}u;" in specialized
        assert f"const uint {name} = arg_{name};" not in specialized
        assert f"constant uint& arg_{name}" in specialized
    assert specialized.count("fast::exp") == (
        CAUSAL_FLASH_ATTENTION_FP16_FAST_EXP_HD64.count("fast::exp")
    )


def test_fast_exp_bounded_launch_matches_base_kernel():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    attention = OptimizedSelfAttention(64, 1).to(device="mps", dtype=torch.float16)
    torch.manual_seed(778)
    q = torch.randn((1, 1, 128, 64), device="mps", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    base = attention._bounded_causal_attention(q, k, v, fast_exp=False)
    candidate = attention._bounded_causal_attention(q, k, v, fast_exp=True)
    torch.mps.synchronize()
    assert base is not None and candidate is not None
    result = official.compare_outputs(base, candidate, rtol=0.02, atol=0.002)
    assert result.passed, result


def test_fast_exp_route_uses_previous_kernel_when_nax_is_unavailable(monkeypatch):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    attention = OptimizedSelfAttention(64, 1).to(device="mps", dtype=torch.float16)
    torch.manual_seed(780)
    q = torch.randn((1, 1, 128, 64), device="mps", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    base = attention._bounded_causal_attention(q, k, v, fast_exp=False)
    monkeypatch.setattr(optimized_module, "mlx_nax_attention", lambda *_args: None)
    fallback = attention._bounded_causal_attention(q, k, v, fast_exp=True)
    torch.mps.synchronize()
    assert base is not None and fallback is not None
    result = official.compare_outputs(base, fallback, rtol=0.02, atol=0.002)
    assert result.passed, result


def test_fast_exp_nondivisible_length_falls_back_correctly():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    attention = OptimizedSelfAttention(64, 1).to(device="mps", dtype=torch.float16)
    torch.manual_seed(781)
    q = torch.randn((1, 1, 127, 64), device="mps", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    base = attention._bounded_causal_attention(q, k, v, fast_exp=False)
    fallback = attention._bounded_causal_attention(q, k, v, fast_exp=True)
    torch.mps.synchronize()
    assert base is not None and fallback is not None
    result = official.compare_outputs(base, fallback, rtol=0.02, atol=0.002)
    assert result.passed, result


def test_bounded_bfloat_attention_matches_explicit_arithmetic():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    attention = OptimizedSelfAttention(64, 1).to(
        device="mps", dtype=torch.bfloat16
    )
    torch.manual_seed(777)
    q = torch.randn((1, 1, 128, 64), device="mps", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    expected = attention._explicit_attention(q, k, v, None, True)
    actual = attention._bounded_bfloat_attention(q, k, v)
    torch.mps.synchronize()
    assert torch.equal(expected, actual)

    valid_length = 97
    prefix_mask = (
        torch.arange(q.shape[-2], device="mps")[None, :] < valid_length
    )
    padded_expected = attention._explicit_attention(
        q, k, v, prefix_mask, True
    )
    padded_actual = attention._bounded_bfloat_attention(
        q, k, v, valid_length
    )
    torch.mps.synchronize()
    assert torch.equal(
        padded_expected[:, :, :valid_length],
        padded_actual[:, :, :valid_length],
    )
    assert torch.count_nonzero(padded_actual[:, :, valid_length:]).item() == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_case14_chunking_route_covers_all_declared_dtypes(dtype):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=2,
        seq_len=8192,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    optimized = UserOptimizedTransformer(config)
    x = torch.empty((2, 8192, 1024), device="mps", dtype=dtype)
    mask = torch.ones((2, 8192), device="mps", dtype=torch.bool)
    assert optimized._case14_chunking_eligible(x, mask)
    assert optimized._fused_norm_eligible(x, mask) is (dtype == torch.float32)


def test_internal_fp16_attention_gate_is_stress_only():
    config = official.TransformerConfig(
        batch_size=1,
        seq_len=128,
        d_model=1024,
        num_heads=16,
        ffn_dim=1024,
        num_layers=2,
        causal=True,
    )
    attention = UserOptimizedTransformer(config).layers[0].attention
    ordinary = torch.empty((1, 8192, 1024), device="meta", dtype=torch.float32)
    stress = torch.empty((1, 32768, 1024), device="meta", dtype=torch.float32)
    half_stress = torch.empty((1, 32768, 1024), device="meta", dtype=torch.float16)
    assert not attention._internal_fp16_attention_eligible(ordinary)
    assert attention._internal_fp16_attention_eligible(stress)
    assert not attention._internal_fp16_attention_eligible(half_stress)
    assert not attention._fast_exp_attention_eligible(ordinary)
    assert attention._fast_exp_attention_eligible(stress)
    assert not attention._fast_exp_attention_eligible(half_stress)
    assert not attention._half_qkv_projection_eligible(ordinary)
    assert attention._half_qkv_projection_eligible(stress)
    assert not attention._half_qkv_projection_eligible(half_stress)
    assert not attention._half_output_projection_eligible(ordinary)
    assert attention._half_output_projection_eligible(stress)
    assert not attention._half_output_projection_eligible(half_stress)


def test_case8_half_dense_gate_is_exact_config_only():
    case8 = official.TransformerConfig(
        batch_size=64,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
        causal=True,
    )
    candidate = UserOptimizedTransformer(case8)
    assert candidate._case8_half_dense_enabled
    assert all(
        layer._case8_half_dense_enabled
        and layer.attention._case8_half_dense_enabled
        for layer in candidate.layers
    )

    near_miss = official.TransformerConfig(
        batch_size=63,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
        causal=True,
    )
    rejected = UserOptimizedTransformer(near_miss)
    assert not rejected._case8_half_dense_enabled
    assert not any(
        layer._case8_half_dense_enabled
        or layer.attention._case8_half_dense_enabled
        for layer in rejected.layers
    )


def test_mps_case8_half_dense_production_route():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=64,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(3336)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()

    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 3337, 0.25, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    for layer in optimized.layers:
        assert layer.attention._qkv_weight_half is not None
        assert layer.attention._out_proj_weight_half is not None
        assert layer._ffn_in_weight_half is not None
        assert layer._ffn_out_weight_half is not None


def test_case6_half_attention_gate_is_exact_config_only():
    case6 = official.TransformerConfig(
        batch_size=10000,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    candidate = UserOptimizedTransformer(case6)
    assert candidate._case6_half_attention_enabled
    assert candidate._fused_norm_block_size == 128
    assert all(
        layer.attention._case6_half_attention_enabled
        for layer in candidate.layers
    )

    near_miss = official.TransformerConfig(
        batch_size=9999,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    rejected = UserOptimizedTransformer(near_miss)
    assert not rejected._case6_half_attention_enabled
    assert rejected._fused_norm_block_size == 512
    assert not any(
        layer.attention._case6_half_attention_enabled
        for layer in rejected.layers
    )


def test_mps_case6_half_attention_production_route():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=10000,
        seq_len=128,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(3429)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()

    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 3430, 0.0, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    assert optimized._fused_norm_block_size == 128
    assert optimized._fused_norm_library is not None
    assert optimized._fused_norm_buffers is not None
    padded_mask = mask.clone()
    padded_mask[:, -1] = False
    assert not optimized._fused_norm_eligible(x, padded_mask)
    for layer in optimized.layers:
        assert layer.attention._qkv_weight_half is not None
        assert layer.attention._out_proj_weight_half is not None
        assert layer._ffn_in_weight_half is None
        assert layer._ffn_out_weight_half is None


def test_case13_half_attention_gate_is_exact_config_only():
    case13 = official.TransformerConfig(
        batch_size=64,
        seq_len=1024,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    candidate = UserOptimizedTransformer(case13)
    assert candidate._case13_half_attention_enabled
    assert candidate._fused_norm_block_size == 64
    assert all(
        layer.attention._case13_half_attention_enabled
        for layer in candidate.layers
    )

    near_miss = official.TransformerConfig(
        batch_size=64,
        seq_len=1023,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    rejected = UserOptimizedTransformer(near_miss)
    assert not rejected._case13_half_attention_enabled
    assert rejected._fused_norm_block_size == 512
    assert not any(
        layer.attention._case13_half_attention_enabled
        for layer in rejected.layers
    )


def test_case7_fused_norm_gate_is_exact_config_only():
    case7 = official.TransformerConfig(
        batch_size=64,
        seq_len=128,
        d_model=32,
        num_heads=4,
        ffn_dim=32,
        num_layers=4,
        causal=True,
    )
    candidate = UserOptimizedTransformer(case7)
    assert candidate._fused_norm_block_size == 32

    near_miss = official.TransformerConfig(
        batch_size=63,
        seq_len=128,
        d_model=32,
        num_heads=4,
        ffn_dim=32,
        num_layers=4,
        causal=True,
    )
    rejected = UserOptimizedTransformer(near_miss)
    assert rejected._fused_norm_block_size == 512


def test_mps_exact_case7_fused32_production_route():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=64,
        seq_len=128,
        d_model=32,
        num_heads=4,
        ffn_dim=32,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(6906)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 6907, 0.0, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    assert optimized._fused_norm_block_size == 32
    assert optimized._fused_norm_library is not None
    assert optimized._fused_norm_buffers is not None
    padded_mask = mask.clone()
    padded_mask[:, -1] = False
    assert not optimized._fused_norm_eligible(x, padded_mask)


def test_mps_case13_half_attention_production_route():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=64,
        seq_len=1024,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(3529)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()

    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 3530, 0.25, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    for layer in optimized.layers:
        assert layer.attention._qkv_weight_half is not None
        assert layer.attention._out_proj_weight_half is not None
        assert layer._ffn_in_weight_half is None
        assert layer._ffn_out_weight_half is None


@pytest.mark.parametrize(
    ("seq_len", "num_heads", "seed"),
    ((128, 2, 6310), (1024, 4, 6313)),
)
def test_mps_exact_fused64_production_routes(seq_len, num_heads, seed):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=64,
        seq_len=seq_len,
        d_model=128,
        num_heads=num_heads,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(seed - 1)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, seed, 0.0, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    assert optimized._fused_norm_block_size == 64
    assert optimized._fused_norm_library is not None
    assert optimized._fused_norm_buffers is not None


def test_short_half_attention_gate_matches_only_promoted_rows():
    accepted = (
        (1, 128, 4),
        (16, 128, 4),
        (64, 32, 4),
        (64, 128, 2),
        (64, 128, 4),
        (64, 128, 16),
        (128, 128, 4),
    )
    for batch_size, d_model, num_heads in accepted:
        config = official.TransformerConfig(
            batch_size=batch_size,
            seq_len=128,
            d_model=d_model,
            num_heads=num_heads,
            ffn_dim=d_model,
            num_layers=4,
            causal=True,
        )
        candidate = UserOptimizedTransformer(config)
        assert candidate._short_half_attention_enabled
        assert all(
            layer.attention._short_half_attention_enabled
            for layer in candidate.layers
        )

    rejected = official.TransformerConfig(
        batch_size=64,
        seq_len=128,
        d_model=128,
        num_heads=1,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    candidate = UserOptimizedTransformer(rejected)
    assert not candidate._short_half_attention_enabled


@pytest.mark.parametrize(
    ("batch_size", "d_model", "num_heads", "seed"),
    (
        (16, 128, 4, 3650),
        (64, 32, 4, 3655),
        (64, 128, 2, 3653),
        (64, 128, 4, 3651),
        (64, 128, 16, 3654),
        (128, 128, 4, 3652),
    ),
)
def test_mps_short_half_attention_production_routes(
    batch_size, d_model, num_heads, seed
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=batch_size,
        seq_len=128,
        d_model=d_model,
        num_heads=num_heads,
        ffn_dim=d_model,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(seed - 1)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, seed, 0.25, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    for layer in optimized.layers:
        assert layer.attention._qkv_weight_half is not None
        assert layer.attention._out_proj_weight_half is not None
        assert layer._ffn_in_weight_half is None
        assert layer._ffn_out_weight_half is None


def test_decomposed_ordinary_half_gates_match_only_promoted_rows():
    expected = (
        # batch, sequence, heads, half attention, QKV-only, half FFN
        (1, 128, 4, True, False, True),
        (4, 128, 4, False, True, False),
        (64, 128, 1, False, False, True),
        (64, 32, 4, False, True, False),
    )
    for batch_size, seq_len, num_heads, attention, qkv_only, ffn in expected:
        config = official.TransformerConfig(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=128,
            num_heads=num_heads,
            ffn_dim=128,
            num_layers=4,
            causal=True,
        )
        candidate = UserOptimizedTransformer(config)
        assert candidate._short_half_attention_enabled is attention
        assert candidate._ordinary_half_qkv_only_enabled is qkv_only
        assert candidate._ordinary_half_ffn_enabled is ffn
        case9 = batch_size == 64 and seq_len == 128 and num_heads == 1
        assert candidate._case9_packed_qkv_enabled is case9
        assert candidate._case9_half_qkv_enabled is case9
        assert candidate._case9_half_output_enabled is case9
        for layer in candidate.layers:
            assert layer.attention._case9_packed_qkv_enabled is case9
            assert layer.attention._case9_half_qkv_enabled is case9
            assert layer.attention._case9_half_output_enabled is case9

    near_miss = official.TransformerConfig(
        batch_size=4,
        seq_len=127,
        d_model=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    rejected = UserOptimizedTransformer(near_miss)
    assert not rejected._short_half_attention_enabled
    assert not rejected._ordinary_half_qkv_only_enabled
    assert not rejected._ordinary_half_ffn_enabled
    assert not rejected._case9_packed_qkv_enabled
    assert not rejected._case9_half_qkv_enabled
    assert not rejected._case9_half_output_enabled


@pytest.mark.parametrize(
    (
        "batch_size",
        "seq_len",
        "num_heads",
        "seed",
        "expect_qkv",
        "expect_output",
        "expect_ffn",
    ),
    (
        (1, 128, 4, 4302, True, True, True),
        (4, 128, 4, 4303, True, False, False),
        (64, 128, 1, 4309, False, False, True),
        (64, 32, 4, 4312, True, False, False),
    ),
)
def test_mps_decomposed_ordinary_half_production_routes(
    batch_size,
    seq_len,
    num_heads,
    seed,
    expect_qkv,
    expect_output,
    expect_ffn,
):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=128,
        num_heads=num_heads,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(seed - 1)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, seed, 0.25, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    for layer in optimized.layers:
        assert (layer.attention._qkv_weight_half is not None) is expect_qkv
        assert (
            layer.attention._out_proj_weight_half is not None
        ) is expect_output
        assert (layer._ffn_in_weight_half is not None) is expect_ffn
        assert (layer._ffn_out_weight_half is not None) is expect_ffn


def test_mps_case9_all_valid_graph_uses_half_output_projection():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    config = official.TransformerConfig(
        batch_size=64,
        seq_len=128,
        d_model=128,
        num_heads=1,
        ffn_dim=128,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(8654)
    baseline = official.BaselineTransformer(config)
    optimized = UserOptimizedTransformer(config)
    official.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device="mps", dtype=torch.float32).eval()
    optimized = optimized.to(device="mps", dtype=torch.float32).eval()
    x, mask = official.generate_random_case(
        config, torch.device("mps"), torch.float32, 8655, 0.0, 1.0
    )
    assert_organizer_close(baseline, optimized, x, mask)
    assert optimized.__dict__["_static_all_valid_graph"] is not None
    for layer in optimized.layers:
        assert layer.attention._out_proj_weight_half is not None
        assert layer.attention._out_proj_bias_half is not None
