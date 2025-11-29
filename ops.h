#ifndef OPS_H
#define OPS_H

#include <iostream>
#include <stdexcept>
#include <vector>

// Forward declarations
struct TensorInfo;
class GGUFFile;

void rms_norm(std::vector<float>& o, const std::vector<float>& x, double eps);
void mat_vec_mul(std::vector<float>& o, const std::vector<float>& w,
                 const std::vector<float>& x);
void softmax(std::vector<float>& x);
void vec_mat_mul(std::vector<float>& o, const std::vector<float>& x,
                 const std::vector<std::vector<float>>& w);

// Quantized matrix-vector multiplication: o = w_q4 * x
// w_tensor contains Q4_0 quantized weights (stored in GGUF file)
// x is F32 input vector
// o is F32 output vector
void mat_vec_mul_q4_0(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x);

// Initialize the operations library (e.g. thread pool)
void init_ops(int n_threads);

#endif  // OPS_H