#ifndef METAL_CONTEXT_H
#define METAL_CONTEXT_H

#import <Metal/Metal.h>

#include <unordered_map>
#include <vector>

#include "gguf.h"

class MetalContext {
 public:
  static MetalContext& get_instance();

  void init();
  bool is_initialized() const { return initialized_; }

  id<MTLDevice> get_device() { return device_; }
  id<MTLCommandQueue> get_command_queue() { return command_queue_; }
  id<MTLComputePipelineState> get_pipeline_state(
      const std::string& kernel_name);

  // Buffer management
  id<MTLBuffer> get_buffer(const TensorInfo& tensor, const GGUFFile& gguf_file);
  id<MTLBuffer> get_temp_buffer(size_t size);

 private:
  MetalContext() = default;
  ~MetalContext() = default;

  bool initialized_ = false;
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> command_queue_ = nil;
  id<MTLLibrary> library_ = nil;

  std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_states_;
  std::unordered_map<const TensorInfo*, id<MTLBuffer>> weight_buffers_;
};

#endif  // METAL_CONTEXT_H
