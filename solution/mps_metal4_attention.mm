#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

#ifdef TRACK3_NAX_TELEMETRY
#include <atomic>
#include <mutex>
#include <vector>
#endif

#include <torch/extension.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

namespace {

id<MTLComputePipelineState> attention_pipeline = nil;

struct AttnParamsHost {
  int32_t B;
  int32_t H;
  int32_t D;
  int32_t qL;
  int32_t kL;
  int32_t gqa_factor;
  float scale;
  int32_t NQ;
  int32_t NK;
  int32_t NQ_aligned;
  int32_t NK_aligned;
  int32_t qL_rem;
  int32_t kL_rem;
  int32_t qL_off;
  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
};

static_assert(sizeof(AttnParamsHost) == 152, "AttnParams ABI drift");

#ifdef TRACK3_NAX_TELEMETRY
struct NAXTelemetryRecord {
  uint64_t launch_id;
  uint64_t buffer_id;
  int64_t status;
  double gpu_ms;
  double kernel_ms;
  std::string error;
};

std::atomic<uint64_t> next_telemetry_launch_id{1};
std::mutex telemetry_mutex;
std::vector<NAXTelemetryRecord> telemetry_records;

void attach_nax_telemetry(id<MTLCommandBuffer> command_buffer) {
  const uint64_t launch_id = next_telemetry_launch_id.fetch_add(1);
  const uint64_t buffer_id =
      static_cast<uint64_t>((uintptr_t)(__bridge void*)command_buffer);
  [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed) {
    NAXTelemetryRecord record{};
    record.launch_id = launch_id;
    record.buffer_id = buffer_id;
    record.status = static_cast<int64_t>(completed.status);
    record.gpu_ms =
        (completed.GPUEndTime - completed.GPUStartTime) * 1000.0;
    record.kernel_ms =
        (completed.kernelEndTime - completed.kernelStartTime) * 1000.0;
    if (completed.error != nil &&
        completed.error.localizedDescription != nil) {
      record.error = completed.error.localizedDescription.UTF8String;
    }
    std::lock_guard<std::mutex> guard(telemetry_mutex);
    telemetry_records.push_back(std::move(record));
  }];
}
#endif

} // namespace

std::string compile_metal4_source(
    const std::string& source,
    const std::string& function_name) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) {
      throw std::runtime_error("MTLCreateSystemDefaultDevice returned nil");
    }

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    if (@available(macOS 26.0, *)) {
      options.languageVersion = MTLLanguageVersion4_0;
    } else {
      throw std::runtime_error("Metal language 4.0 requires macOS 26 or newer");
    }

    NSError* error = nil;
    NSString* metal_source =
        [[NSString alloc] initWithBytes:source.data()
                                length:source.size()
                              encoding:NSUTF8StringEncoding];
    id<MTLLibrary> library = [device newLibraryWithSource:metal_source
                                                  options:options
                                                    error:&error];
    if (library == nil) {
      std::string message = "Metal 4 compilation failed";
      if (error != nil && error.localizedDescription != nil) {
        message += ": ";
        message += error.localizedDescription.UTF8String;
      }
      throw std::runtime_error(message);
    }

    NSString* requested_name =
        [[NSString alloc] initWithBytes:function_name.data()
                                  length:function_name.size()
                                encoding:NSUTF8StringEncoding];
    id<MTLFunction> function = [library newFunctionWithName:requested_name];
    if (function == nil) {
      throw std::runtime_error("compiled Metal library lacks requested function");
    }
    NSError* pipeline_error = nil;
    attention_pipeline =
        [device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (attention_pipeline == nil) {
      std::string message = "Metal pipeline creation failed";
      if (pipeline_error != nil && pipeline_error.localizedDescription != nil) {
        message += ": ";
        message += pipeline_error.localizedDescription.UTF8String;
      }
      throw std::runtime_error(message);
    }
    return std::string("ok:") + std::to_string(library.functionNames.count) +
        ":max_threads=" +
        std::to_string(attention_pipeline.maxTotalThreadsPerThreadgroup);
  }
}

at::Tensor run_nax_attention(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    double scale) {
  TORCH_CHECK(attention_pipeline != nil, "compile_metal4_source must run first");
  TORCH_CHECK(q.is_mps() && k.is_mps() && v.is_mps(), "Q/K/V must be MPS tensors");
  TORCH_CHECK(q.scalar_type() == at::kHalf, "Q must be float16");
  TORCH_CHECK(k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf,
              "K/V must be float16");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "Q/K/V must have shape [B,H,L,D]");
  TORCH_CHECK(k.sizes() == v.sizes(), "K/V sizes must match");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1) &&
                  q.size(3) == k.size(3),
              "Q and K/V batch, head, and dimension must match");
  TORCH_CHECK(q.size(2) >= k.size(2) && q.size(2) - k.size(2) < 256,
              "Q may contain fewer than 256 trailing pad rows");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "Q/K/V must be contiguous");
  TORCH_CHECK(q.size(3) == 64, "this probe is specialized to head dimension 64");
  TORCH_CHECK(q.size(2) > 0 && q.size(2) % 256 == 0,
              "Q length must be a positive multiple of 256");
  TORCH_CHECK(k.size(2) > 0 && k.size(2) % 32 == 0,
              "K/V length must be a positive multiple of 32");

  auto output = at::empty_like(q);
  const int64_t batch = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t q_length = q.size(2);
  const int64_t kv_length = k.size(2);
  const int32_t bq = 256;
#ifndef TRACK3_NAX_BK
  // Historical experiment bridges compile without a definition and retain BK32.
  const int32_t bk = 32;
#else
  const int32_t bk = TRACK3_NAX_BK;
#endif

  AttnParamsHost params{};
  params.B = static_cast<int32_t>(batch);
  params.H = static_cast<int32_t>(heads);
  params.D = 64;
  params.qL = static_cast<int32_t>(q_length);
  params.kL = static_cast<int32_t>(kv_length);
  params.gqa_factor = 1;
  params.scale = static_cast<float>(scale);
  params.NQ = static_cast<int32_t>((q_length + bq - 1) / bq);
  params.NK = static_cast<int32_t>((kv_length + bk - 1) / bk);
  params.NQ_aligned = static_cast<int32_t>(q_length / bq);
  params.NK_aligned = static_cast<int32_t>(kv_length / bk);
  params.qL_rem = 0;
#ifndef TRACK3_NAX_BK
  params.kL_rem = 0;
#else
  params.kL_rem = static_cast<int32_t>(
      kv_length - static_cast<int64_t>(params.NK_aligned) * bk);
#endif
  params.qL_off = 0;
  for (int index = 0; index < 3; ++index) {
    params.Q_strides[index] = q.stride(index);
    params.K_strides[index] = k.stride(index);
    params.V_strides[index] = v.stride(index);
    params.O_strides[index] = output.stride(index);
  }

  id<MTLBuffer> q_buffer = at::native::mps::getMTLBufferStorage(q);
  id<MTLBuffer> k_buffer = at::native::mps::getMTLBufferStorage(k);
  id<MTLBuffer> v_buffer = at::native::mps::getMTLBufferStorage(v);
  id<MTLBuffer> o_buffer = at::native::mps::getMTLBufferStorage(output);
  const NSUInteger q_offset = q.storage_offset() * q.element_size();
  const NSUInteger k_offset = k.storage_offset() * k.element_size();
  const NSUInteger v_offset = v.storage_offset() * v.element_size();
  const NSUInteger o_offset = output.storage_offset() * output.element_size();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:attention_pipeline];
    [encoder setBuffer:q_buffer offset:q_offset atIndex:0];
    [encoder setBuffer:k_buffer offset:k_offset atIndex:1];
    [encoder setBuffer:v_buffer offset:v_offset atIndex:2];
    [encoder setBuffer:o_buffer offset:o_offset atIndex:3];
    [encoder setBytes:&params length:sizeof(params) atIndex:4];
    MTLSize grid = MTLSizeMake(params.NQ, params.H, params.B);
    MTLSize group = MTLSizeMake(32, 16, 1);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
#ifdef TRACK3_NAX_TELEMETRY
    MPSCommandBuffer* mps_buffer = stream->commandBuffer();
    attach_nax_telemetry(mps_buffer.rootCommandBuffer);
#endif
  });
  return output;
}

#ifdef TRACK3_NAX_TELEMETRY
pybind11::list take_nax_telemetry() {
  std::lock_guard<std::mutex> guard(telemetry_mutex);
  pybind11::list result;
  for (const NAXTelemetryRecord& record : telemetry_records) {
    pybind11::dict item;
    item["launch_id"] = record.launch_id;
    item["buffer_id"] = record.buffer_id;
    item["status"] = record.status;
    item["gpu_ms"] = record.gpu_ms;
    item["kernel_ms"] = record.kernel_ms;
    item["error"] = record.error;
    result.append(item);
  }
  telemetry_records.clear();
  return result;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "compile_metal4_source",
      &compile_metal4_source,
      "Compile an in-memory Metal 4 source string and report function count");
  module.def(
      "run_nax_attention",
      &run_nax_attention,
      "Run the verified half/BQ256/BK48/BD64/WM16/WN1 NAX attention route");
#ifdef TRACK3_NAX_TELEMETRY
  module.def(
      "take_nax_telemetry",
      &take_nax_telemetry,
      "Return and clear completed NAX command-buffer timestamp records");
#endif
}
