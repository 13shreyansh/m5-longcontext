"""Audited Metal kernels used by the optimized Transformer.

The fused residual-plus-LayerNorm kernel is adapted from triton-msl's
MIT-licensed `make_fused_residual_norm_kernel` output at commit
182c1820fd24a836d565e1da842f28414de64084. The retained licence is in
`solution/third_party/triton-msl-LICENSE`.

The width-32/32-thread and width-128/64-thread/128-thread sources are exact
generator output. The width-1024, 512-thread source is specialized to float32
and LayerNorm epsilon 1e-5 and removes one unused threadgroup array.
"""


FUSED_RESIDUAL_NORM_FP32_32 = r"""
#include <metal_stdlib>
using namespace metal;

kernel void fused_residual_norm(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    volatile device float* output [[buffer(4)]],
    volatile device float* residual_out [[buffer(5)]],
    constant uint& n_cols [[buffer(6)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_sum[1];
    threadgroup float shared_var[1];
    threadgroup float tg_x[32];
    uint row_start = pid * n_cols;
    float local_sum = 0.0f;
    for (uint i = lid; i < n_cols; i += 32u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        residual_out[row_start + i] = x_val;
        local_sum += x_val;
    }
    float simd_total_sum = simd_sum((float)local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 1u) {
        shared_sum[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_sum[sgitg] = simd_total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_sum = (tiisg < 1u) ? shared_sum[tiisg] : 0.0f;
    float total_sum = simd_sum(shared_total_sum);
    if (lid == 0) {
        shared_sum[0] = total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean_val = shared_sum[0] / float(n_cols);
    float local_var = 0.0f;
    for (uint i = lid; i < n_cols; i += 32u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        float diff = x_val - mean_val;
        local_var += diff * diff;
    }
    float simd_total_var = simd_sum((float)local_var);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 1u) {
        shared_var[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_var[sgitg] = simd_total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_var = (tiisg < 1u) ? shared_var[tiisg] : 0.0f;
    float total_var = simd_sum(shared_total_var);
    if (lid == 0) {
        shared_var[0] = total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var_val = shared_var[0] / float(n_cols);
    float inv_std = rsqrt(var_val + 1e-05f);
    for (uint i = lid; i < n_cols; i += 32u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        output[row_start + i] = (x_val - mean_val) * inv_std * gamma[i] + beta[i];
    }
}
"""


FUSED_RESIDUAL_NORM_FP32_128 = r"""
#include <metal_stdlib>
using namespace metal;

kernel void fused_residual_norm(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    volatile device float* output [[buffer(4)]],
    volatile device float* residual_out [[buffer(5)]],
    constant uint& n_cols [[buffer(6)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_sum[4];
    threadgroup float shared_var[4];
    threadgroup float tg_x[128];
    uint row_start = pid * n_cols;
    float local_sum = 0.0f;
    for (uint i = lid; i < n_cols; i += 128u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        residual_out[row_start + i] = x_val;
        local_sum += x_val;
    }
    float simd_total_sum = simd_sum((float)local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 4u) {
        shared_sum[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_sum[sgitg] = simd_total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_sum = (tiisg < 4u) ? shared_sum[tiisg] : 0.0f;
    float total_sum = simd_sum(shared_total_sum);
    if (lid == 0) {
        shared_sum[0] = total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean_val = shared_sum[0] / float(n_cols);
    float local_var = 0.0f;
    for (uint i = lid; i < n_cols; i += 128u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        float diff = x_val - mean_val;
        local_var += diff * diff;
    }
    float simd_total_var = simd_sum((float)local_var);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 4u) {
        shared_var[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_var[sgitg] = simd_total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_var = (tiisg < 4u) ? shared_var[tiisg] : 0.0f;
    float total_var = simd_sum(shared_total_var);
    if (lid == 0) {
        shared_var[0] = total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var_val = shared_var[0] / float(n_cols);
    float inv_std = rsqrt(var_val + 1e-05f);
    for (uint i = lid; i < n_cols; i += 128u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        output[row_start + i] = (x_val - mean_val) * inv_std * gamma[i] + beta[i];
    }
}
"""


FUSED_RESIDUAL_NORM_FP32_64 = r"""
#include <metal_stdlib>
using namespace metal;

kernel void fused_residual_norm(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    volatile device float* output [[buffer(4)]],
    volatile device float* residual_out [[buffer(5)]],
    constant uint& n_cols [[buffer(6)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tid [[thread_position_in_grid]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_sum[2];
    threadgroup float shared_var[2];
    threadgroup float tg_x[64];
    uint row_start = pid * n_cols;
    float local_sum = 0.0f;
    for (uint i = lid; i < n_cols; i += 64u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        residual_out[row_start + i] = x_val;
        local_sum += x_val;
    }
    float simd_total_sum = simd_sum((float)local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 2u) {
        shared_sum[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_sum[sgitg] = simd_total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_sum = (tiisg < 2u) ? shared_sum[tiisg] : 0.0f;
    float total_sum = simd_sum(shared_total_sum);
    if (lid == 0) {
        shared_sum[0] = total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean_val = shared_sum[0] / float(n_cols);
    float local_var = 0.0f;
    for (uint i = lid; i < n_cols; i += 64u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        float diff = x_val - mean_val;
        local_var += diff * diff;
    }
    float simd_total_var = simd_sum((float)local_var);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 2u) {
        shared_var[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_var[sgitg] = simd_total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_var = (tiisg < 2u) ? shared_var[tiisg] : 0.0f;
    float total_var = simd_sum(shared_total_var);
    if (lid == 0) {
        shared_var[0] = total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var_val = shared_var[0] / float(n_cols);
    float inv_std = rsqrt(var_val + 1e-05f);
    for (uint i = lid; i < n_cols; i += 64u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        output[row_start + i] = (x_val - mean_val) * inv_std * gamma[i] + beta[i];
    }
}
"""


FUSED_RESIDUAL_NORM_FP32_512 = r"""
#include <metal_stdlib>
using namespace metal;

kernel void fused_residual_norm(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* gamma [[buffer(2)]],
    device const float* beta [[buffer(3)]],
    volatile device float* output [[buffer(4)]],
    volatile device float* residual_out [[buffer(5)]],
    constant uint& n_cols [[buffer(6)]],
    uint pid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_sum[16];
    threadgroup float shared_var[16];
    uint row_start = pid * n_cols;
    float local_sum = 0.0f;
    for (uint i = lid; i < n_cols; i += 512u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        residual_out[row_start + i] = x_val;
        local_sum += x_val;
    }
    float simd_total_sum = simd_sum(local_sum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 16u) {
        shared_sum[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_sum[sgitg] = simd_total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_sum = (tiisg < 16u) ? shared_sum[tiisg] : 0.0f;
    float total_sum = simd_sum(shared_total_sum);
    if (lid == 0) {
        shared_sum[0] = total_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean_val = shared_sum[0] / float(n_cols);

    float local_var = 0.0f;
    for (uint i = lid; i < n_cols; i += 512u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        float diff = x_val - mean_val;
        local_var += diff * diff;
    }
    float simd_total_var = simd_sum(local_var);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg < 16u) {
        shared_var[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shared_var[sgitg] = simd_total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float shared_total_var = (tiisg < 16u) ? shared_var[tiisg] : 0.0f;
    float total_var = simd_sum(shared_total_var);
    if (lid == 0) {
        shared_var[0] = total_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var_val = shared_var[0] / float(n_cols);
    float inv_std = rsqrt(var_val + 1e-05f);

    for (uint i = lid; i < n_cols; i += 512u) {
        float x_val = input[row_start + i] + residual[row_start + i];
        output[row_start + i] =
            (x_val - mean_val) * inv_std * gamma[i] + beta[i];
    }
}
"""
