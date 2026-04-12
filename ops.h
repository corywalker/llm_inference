#ifndef OPS_H
#define OPS_H

#include <vector>

#include "tensor.h"

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q4_K {
  uint16_t d;                    // super-block scale for quantized scales
  uint16_t dmin;                 // super-block scale for quantized mins
  uint8_t scales[K_SCALE_SIZE];  // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2];          // 4-bit quants
};

struct block_q6_K {
  uint8_t ql[QK_K / 2];      // quants, lower 4 bits
  uint8_t qh[QK_K / 4];      // quants, upper 2 bits
  int8_t scales[QK_K / 16];  // scales, quantized with 8 bits
  uint16_t d;                // super-block scale
};

#pragma pack(push, 1)
struct block_q5_0 {
  uint16_t d;     // delta
  uint8_t qh[4];  // 5-th bit of quants
  uint8_t qs[16]; // low 4 bits of quants
};
#pragma pack(pop)

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
// w_tensor contains quantized weights (stored in GGUF file)
// x is F32 input vector
// o is F32 output vector
void mat_vec_mul(std::vector<float>& o, const TensorInfo& w_tensor,
                 const GGUFFile& gguf_file, const std::vector<float>& x);

void mat_vec_mul_q4_0(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x);

// Quantized matrix-vector multiplication for Q4_K and Q6_K weights
void mat_vec_mul_q4_k(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x);

void mat_vec_mul_q6_k(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x);
void mat_vec_mul_q8_0(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x);
void mat_vec_mul_q5_0(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x);

void dequantize_q6_k_row(std::vector<float>& o, const uint8_t* block_ptr,
                         size_t n_cols);
void dequantize_q8_0_row(std::vector<float>& o, const uint8_t* block_ptr,
                         size_t n_cols);
void dequantize_q5_0_row(std::vector<float>& o, const uint8_t* block_ptr,
                         size_t n_cols);

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