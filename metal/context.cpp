#include "context.h"
#include <iostream>
#include <fstream>

MetalContext& MetalContext::get_instance() {
    static MetalContext instance;
    return instance;
}

void MetalContext::init() {
    if (initialized_) return;

    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "Error: Failed to create Metal device." << std::endl;
        return;
    }

    command_queue_ = [device_ newCommandQueue];
    if (!command_queue_) {
        std::cerr << "Error: Failed to create Metal command queue." << std::endl;
        return;
    }

    // Load library from source (since we are not bundling a .metallib yet)
    // In a real app, we might want to compile this offline.
    // For now, we'll read the source file.
    // Assuming the kernels.metal file is in the same directory as the binary or accessible.
    // Since we are running with bazel, data dependencies are tricky.
    // Let's try to find it relative to the binary or embedded.
    // For simplicity in this script-like environment, let's assume it's at "metal/kernels.metal"
    // relative to the working directory.
    
    NSError* error = nil;
    NSString* source = [NSString stringWithContentsOfFile:@"metal/kernels.metal"
                                                 encoding:NSUTF8StringEncoding
                                                    error:&error];
    if (!source) {
        // Fallback: try to read from the runfiles location if possible, or just hardcode for now?
        // Let's try to read from the absolute path if we can't find it relative.
        // Actually, for this environment, let's just assume we run from the workspace root.
        std::cerr << "Error: Failed to load metal/kernels.metal: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    library_ = [device_ newLibraryWithSource:source options:options error:&error];
    if (!library_) {
        std::cerr << "Error: Failed to compile Metal library: " << [[error localizedDescription] UTF8String] << std::endl;
        return;
    }

    initialized_ = true;
    std::cout << "Metal initialized successfully." << std::endl;
}

id<MTLComputePipelineState> MetalContext::get_pipeline_state(const std::string& kernel_name) {
    if (pipeline_states_.find(kernel_name) != pipeline_states_.end()) {
        return pipeline_states_[kernel_name];
    }

    NSString* name = [NSString stringWithUTF8String:kernel_name.c_str()];
    id<MTLFunction> function = [library_ newFunctionWithName:name];
    if (!function) {
        std::cerr << "Error: Failed to find function " << kernel_name << std::endl;
        return nil;
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pipeline_state = [device_ newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline_state) {
        std::cerr << "Error: Failed to create pipeline state for " << kernel_name << ": " << [[error localizedDescription] UTF8String] << std::endl;
        return nil;
    }

    pipeline_states_[kernel_name] = pipeline_state;
    return pipeline_state;
}

id<MTLBuffer> MetalContext::get_buffer(const TensorInfo& tensor, const GGUFFile& gguf_file) {
    if (weight_buffers_.find(&tensor) != weight_buffers_.end()) {
        return weight_buffers_[&tensor];
    }

    // Calculate size
    // For Q4_0, we need the raw bytes.
    // tensor.shape[0] is cols (embedding dim)
    // tensor.shape[1] is rows (output dim)
    size_t n_cols = tensor.shape[0];
    size_t n_rows = tensor.shape[1];
    
    // Q4_0: 32 values -> 18 bytes
    size_t block_size = 32;
    size_t bytes_per_block = 18;
    size_t blocks_per_row = (n_cols + block_size - 1) / block_size;
    size_t total_bytes = n_rows * blocks_per_row * bytes_per_block;

    // Get data pointer
    const uint8_t* data = gguf_file.get_tensor_data(tensor);

    // Create buffer
    id<MTLBuffer> buffer = [device_ newBufferWithBytes:data
                                                length:total_bytes
                                               options:MTLResourceStorageModeShared];
    
    weight_buffers_[&tensor] = buffer;
    return buffer;
}

id<MTLBuffer> MetalContext::get_temp_buffer(size_t size) {
    return [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
}
