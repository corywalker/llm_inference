#include "ops.h"
#include "context.h"
#include <iostream>

void init_metal() {
    MetalContext::get_instance().init();
}

struct MetalTensorInfo {
    int rows;
    int cols;
};

void metal_mat_vec_mul_q4_0(std::vector<float>& o, const TensorInfo& w_tensor,
                            const GGUFFile& gguf_file, const std::vector<float>& x) {
    auto& ctx = MetalContext::get_instance();
    if (!ctx.is_initialized()) {
        std::cerr << "Error: Metal not initialized." << std::endl;
        return;
    }

    size_t n_cols = w_tensor.shape[0];
    size_t n_rows = w_tensor.shape[1];

    if (x.size() != n_cols) {
        std::cerr << "Error: Input size mismatch in metal_mat_vec_mul_q4_0" << std::endl;
        return;
    }

    o.resize(n_rows);

    id<MTLComputePipelineState> pso = ctx.get_pipeline_state("mat_vec_mul_q4_0");
    if (!pso) return;

    id<MTLCommandBuffer> commandBuffer = [ctx.get_command_queue() commandBuffer];
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    [computeEncoder setComputePipelineState:pso];

    // Buffer 0: Weights
    id<MTLBuffer> w_buffer = ctx.get_buffer(w_tensor, gguf_file);
    [computeEncoder setBuffer:w_buffer offset:0 atIndex:0];

    // Buffer 1: Input
    id<MTLBuffer> x_buffer = ctx.get_temp_buffer(x.size() * sizeof(float));
    memcpy(x_buffer.contents, x.data(), x.size() * sizeof(float));
    [computeEncoder setBuffer:x_buffer offset:0 atIndex:1];

    // Buffer 2: Output
    id<MTLBuffer> o_buffer = ctx.get_temp_buffer(n_rows * sizeof(float));
    [computeEncoder setBuffer:o_buffer offset:0 atIndex:2];

    // Buffer 3: Info
    MetalTensorInfo info = {(int)n_rows, (int)n_cols};
    [computeEncoder setBytes:&info length:sizeof(info) atIndex:3];

    // Dispatch
    // One thread per row
    MTLSize gridSize = MTLSizeMake(n_rows, 1, 1);
    
    // Threadgroup size: try to use 32 (simd width) or PSO max
    NSUInteger threadGroupSize = std::min((NSUInteger)32, pso.maxTotalThreadsPerThreadgroup);
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [computeEncoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy back
    memcpy(o.data(), o_buffer.contents, n_rows * sizeof(float));
}
