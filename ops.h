#ifndef OPS_H
#define OPS_H

#include <vector>

#include "tensor.h"

// Forward declarations
struct TensorInfo;
class GGUFFile;

// Initialize the operations library (e.g. thread pool)
void init_ops(int n_threads);

// Matrix-Vector Multiplication with FP16 weights
void mat_vec_mul_fp16(std::vector<float>& o, const std::vector<uint16_t>& w,
                      const std::vector<float>& x, size_t n_rows,
                      size_t n_cols);

// FP16 Vector Operations
void vec_scale_f16(tensor_f16_1& y, float v);
void vec_mad_f16(tensor_f16_1& y, const tensor_f16_1& x, float v);

// Quantized matrix-vector multiplication: o = w_q4 * x
// w_tensor contains Q4_0 quantized weights (stored in GGUF file)
// x is F32 input vector
// o is F32 output vector
void mat_vec_mul_q4_0(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x);

// Misc operations.
void rms_norm(std::vector<float>& o, const std::vector<float>& x, double eps);
void softmax(std::vector<float>& x);
void rope(tensor_3& tensor, int n_rot, float rope_freq_base,
          float rope_freq_scale, int pos);
void scale(tensor_3& tensor, float scale_factor);

// Q8_0 quantization block for activations
struct BlockQ8_0 {
  uint16_t d;     // scaling factor (fp16)
  int8_t qs[32];  // quantized values
};

void quantize_row_q8_0(const std::vector<float>& x, std::vector<BlockQ8_0>& y,
                       size_t size);

#endif  // OPS_H