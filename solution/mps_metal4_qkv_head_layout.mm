// Fail-closed row-14 packed-QKV head-major output bridge.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdexcept>
#include <string>

#include <torch/extension.h>

#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#ifndef TRACK3_LINEAR_M
#define TRACK3_LINEAR_M 100000
#endif
#ifndef TRACK3_LINEAR_BM
#define TRACK3_LINEAR_BM 64
#endif
#ifndef TRACK3_LINEAR_BN
#define TRACK3_LINEAR_BN 256
#endif
#ifndef TRACK3_LINEAR_WM
#define TRACK3_LINEAR_WM 2
#endif
#ifndef TRACK3_LINEAR_WN
#define TRACK3_LINEAR_WN 4
#endif

namespace {
id<MTLComputePipelineState> qkv_pipeline = nil;
}

std::string compile_qkv_head_layout_source(
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
      std::string message = "Metal 4 QKV compilation failed";
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
      throw std::runtime_error("compiled library lacks requested QKV function");
    }
    NSError* pipeline_error = nil;
    qkv_pipeline =
        [device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (qkv_pipeline == nil) {
      std::string message = "Metal QKV pipeline creation failed";
      if (pipeline_error != nil && pipeline_error.localizedDescription != nil) {
        message += ": ";
        message += pipeline_error.localizedDescription.UTF8String;
      }
      throw std::runtime_error(message);
    }
    return std::string("ok:") + std::to_string(library.functionNames.count) +
        ":max_threads=" +
        std::to_string(qkv_pipeline.maxTotalThreadsPerThreadgroup);
  }
}

at::Tensor run_qkv_head_layout(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& bias) {
  TORCH_CHECK(qkv_pipeline != nil,
              "compile_qkv_head_layout_source must run first");
  TORCH_CHECK(x.is_mps() && weight.is_mps() && bias.is_mps(),
              "X/weight/bias must be MPS tensors");
  TORCH_CHECK(x.scalar_type() == at::kHalf &&
                  weight.scalar_type() == at::kHalf &&
                  bias.scalar_type() == at::kHalf,
              "X/weight/bias must be float16");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 1024,
              "X must have shape [padded_M,1024]");
  TORCH_CHECK(x.size(0) >= TRACK3_LINEAR_M &&
                  x.size(0) < TRACK3_LINEAR_M + TRACK3_LINEAR_BM &&
                  x.size(0) % TRACK3_LINEAR_BM == 0,
              "X must contain fewer than one BM block of trailing pad rows");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({3072, 1024}),
              "weight must have shape [3072,1024]");
  TORCH_CHECK(bias.sizes() == at::IntArrayRef({3072}),
              "bias must have shape [3072]");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(),
              "X/weight/bias must be contiguous");

  auto output = at::empty({48, TRACK3_LINEAR_M, 64}, x.options());
  id<MTLBuffer> x_buffer = at::native::mps::getMTLBufferStorage(x);
  id<MTLBuffer> w_buffer = at::native::mps::getMTLBufferStorage(weight);
  id<MTLBuffer> b_buffer = at::native::mps::getMTLBufferStorage(bias);
  id<MTLBuffer> y_buffer = at::native::mps::getMTLBufferStorage(output);
  const NSUInteger x_offset = x.storage_offset() * x.element_size();
  const NSUInteger w_offset = weight.storage_offset() * weight.element_size();
  const NSUInteger b_offset = bias.storage_offset() * bias.element_size();
  const NSUInteger y_offset = output.storage_offset() * output.element_size();
  at::mps::MPSStream* stream = at::mps::getCurrentMPSStream();
  const NSUInteger tiles_m =
      static_cast<NSUInteger>((TRACK3_LINEAR_M + TRACK3_LINEAR_BM - 1) /
                              TRACK3_LINEAR_BM);

  dispatch_sync(stream->queue(), ^{
    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
    [encoder setComputePipelineState:qkv_pipeline];
    [encoder setBuffer:x_buffer offset:x_offset atIndex:0];
    [encoder setBuffer:w_buffer offset:w_offset atIndex:1];
    [encoder setBuffer:b_buffer offset:b_offset atIndex:2];
    [encoder setBuffer:y_buffer offset:y_offset atIndex:3];
    MTLSize grid = MTLSizeMake(3072 / TRACK3_LINEAR_BN, tiles_m, 1);
    MTLSize group =
        MTLSizeMake(32, TRACK3_LINEAR_WN, TRACK3_LINEAR_WM);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:group];
  });
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "compile_qkv_head_layout_source",
      &compile_qkv_head_layout_source,
      "Compile direct packed-QKV head-layout Metal source");
  module.def(
      "run_qkv_head_layout",
      &run_qkv_head_layout,
      "Run packed QKV and return [48,M,64] head-major storage");
}
