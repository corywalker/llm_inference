#include "ops.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "gguf.h"
#include "thread_pool.h"
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#if defined(__aarch64__) || defined(__arm64__)
#include <arm_neon.h>
#endif

static std::unique_ptr<ThreadPool> pool;
static int g_n_threads = 1;

void init_ops(int n_threads) {
  g_n_threads = n_threads;
  pool = std::make_unique<ThreadPool>(n_threads);
}

// GGML implementation appears to be at:
// https://github.com/ggml-org/llama.cpp/blob/75cbdd3fce38ea12d50cd19e73a069aa5dbbd5fa/ggml/src/ggml-cpu/ops.cpp#L3517
void rms_norm(std::vector<float>& o, const std::vector<float>& x, double eps) {
  if (eps <= 0) {
    std::cerr << "Error: eps must be > 0 in rms_norm." << std::endl;
    exit(1);
  }
  float sum = 0.0f;
  for (float val : x) {
    sum += val * val;
  }
  const float mean = sum / x.size();
  const float scale = 1.0f / sqrtf(mean + eps);

  for (size_t i = 0; i < o.size(); ++i) {
    o[i] = scale * x[i];
  }
}

void mat_vec_mul(std::vector<float>& o, const std::vector<float>& w,
                 const std::vector<float>& x) {
  size_t C = x.size();
  if (C == 0) {
    o.clear();
    return;
  }
  if (w.size() % C != 0) {
    throw std::runtime_error("mat_vec_mul: dimensions mismatch");
  }
  size_t R = w.size() / C;
  o.resize(R);

  auto compute_range = [&](size_t start_row, size_t end_row) {
#if defined(__x86_64__)
    for (size_t i = start_row; i < end_row; ++i) {
      const float* w_ptr = w.data() + i * C;
      const float* x_ptr = x.data();
      __m256 sum_vec = _mm256_setzero_ps();
      size_t j = 0;
      for (; j + 7 < C; j += 8) {
        __m256 w_vec = _mm256_loadu_ps(w_ptr + j);
        __m256 x_vec = _mm256_loadu_ps(x_ptr + j);
        sum_vec = _mm256_fmadd_ps(w_vec, x_vec, sum_vec);
      }

      float sum_arr[8];
      _mm256_storeu_ps(sum_arr, sum_vec);
      float result = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                     sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];

      for (; j < C; ++j) {
        result += w_ptr[j] * x_ptr[j];
      }
      o[i] = result;
    }
#elif defined(__aarch64__) || defined(__arm64__)
    for (size_t i = start_row; i < end_row; ++i) {
      const float* w_ptr = w.data() + i * C;
      const float* x_ptr = x.data();

      // Use 4 accumulators to break dependency chains
      float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
      float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
      float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
      float32x4_t sum_vec3 = vdupq_n_f32(0.0f);

      size_t j = 0;
      // Process 16 floats per iteration (4x unroll)
      for (; j + 15 < C; j += 16) {
        float32x4_t w_vec0 = vld1q_f32(w_ptr + j);
        float32x4_t x_vec0 = vld1q_f32(x_ptr + j);
        sum_vec0 = vfmaq_f32(sum_vec0, w_vec0, x_vec0);

        float32x4_t w_vec1 = vld1q_f32(w_ptr + j + 4);
        float32x4_t x_vec1 = vld1q_f32(x_ptr + j + 4);
        sum_vec1 = vfmaq_f32(sum_vec1, w_vec1, x_vec1);

        float32x4_t w_vec2 = vld1q_f32(w_ptr + j + 8);
        float32x4_t x_vec2 = vld1q_f32(x_ptr + j + 8);
        sum_vec2 = vfmaq_f32(sum_vec2, w_vec2, x_vec2);

        float32x4_t w_vec3 = vld1q_f32(w_ptr + j + 12);
        float32x4_t x_vec3 = vld1q_f32(x_ptr + j + 12);
        sum_vec3 = vfmaq_f32(sum_vec3, w_vec3, x_vec3);
      }

      // Combine accumulators
      sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
      sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
      sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);

      // Handle remaining 4-wide chunks
      for (; j + 3 < C; j += 4) {
        float32x4_t w_vec = vld1q_f32(w_ptr + j);
        float32x4_t x_vec = vld1q_f32(x_ptr + j);
        sum_vec0 = vfmaq_f32(sum_vec0, w_vec, x_vec);
      }

      float result = vaddvq_f32(sum_vec0);

      for (; j < C; ++j) {
        result += w_ptr[j] * x_ptr[j];
      }
      o[i] = result;
    }
#else
    for (size_t i = start_row; i < end_row; ++i) {
      o[i] = 0.0f;
      for (size_t j = 0; j < C; ++j) {
        o[i] += w[i * C + j] * x[j];
      }
    }
#endif
  };

  std::vector<std::future<void>> results;
  size_t num_threads = g_n_threads;
  size_t chunk_size = (R + num_threads - 1) / num_threads;

  for (size_t i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end = std::min(start + chunk_size, R);
    if (start >= end) break;
    results.emplace_back(pool->enqueue(compute_range, start, end));
  }

  for (auto&& result : results) result.get();
}

void softmax(std::vector<float>& x) {
  float max_val = x[0];
  for (float val : x) {
    if (val > max_val) {
      max_val = val;
    }
  }

  float sum = 0.0f;
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }

  for (size_t i = 0; i < x.size(); ++i) {
    x[i] /= sum;
  }
}

// Refer to
// https://github.com/ggml-org/llama.cpp/blob/1f5accb8d0056e6099cd5b772b1cb787dd590a13/ggml/src/ggml-cpu/ops.cpp#L5546C13-L5546C42
// as a golden implementation.
void rope(tensor_3& tensor, int n_rot, float rope_freq_base,
          float rope_freq_scale, int pos) {
  uint32_t n_tokens = tensor.size();
  if (n_tokens == 0) {
    return;
  }
  uint32_t n_heads = tensor[0].size();
  if (n_heads == 0) {
    return;
  }

  for (uint32_t t = 0; t < n_tokens; ++t) {
    for (int i = 0; i < n_rot / 2; ++i) {
      const float freq = 1.0f / powf(rope_freq_base, (float)(2 * i) / n_rot);
      const float val = (pos + t) * freq / rope_freq_scale;
      const float fcr = cosf(val);
      const float fci = sinf(val);

      for (uint32_t h = 0; h < n_heads; ++h) {
        tensor_1& vec = tensor[t][h];
        const float v0 = vec[i];
        const float v1 = vec[i + n_rot / 2];

        vec[i] = v0 * fcr - v1 * fci;
        vec[i + n_rot / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}

void scale(tensor_3& tensor, float scale_factor) {
  for (auto& token_tensor : tensor) {
    for (auto& head_tensor : token_tensor) {
      for (float& val : head_tensor) {
        val *= scale_factor;
      }
    }
  }
}

// Quantized matrix-vector multiplication for Q4_0 weights
// Reference: llama.cpp ggml_vec_dot_q4_0_q8_0
void mat_vec_mul_q4_0(std::vector<float>& o, const TensorInfo& w_tensor,
                      const GGUFFile& gguf_file, const std::vector<float>& x) {
  // Matrix dimensions: w_tensor is [rows x cols] in Q4_0 format
  // w_tensor.shape[0] = embedding dimension (cols)
  // w_tensor.shape[1] = output dimension (rows)
  const size_t n_rows = w_tensor.shape[1];
  const size_t n_cols = w_tensor.shape[0];

  if (x.size() != n_cols) {
    throw std::runtime_error("mat_vec_mul_q4_0: input vector size mismatch");
  }

  o.resize(n_rows);

  // Q4_0 format: blocks of 32 elements
  // Each block: 2 bytes (float16 scale) + 16 bytes (32 x 4-bit values packed)
  const size_t block_size = 32;
  const size_t bytes_per_block = 2 + 16;  // 2 bytes scale + 16 bytes quants
  const size_t blocks_per_row = (n_cols + block_size - 1) / block_size;
  const uint8_t* w_data = gguf_file.get_tensor_data(w_tensor);

  auto compute_range = [&](size_t start_row, size_t end_row) {
#if defined(__aarch64__) || defined(__arm64__)
    // Process each output row
    for (size_t row = start_row; row < end_row; ++row) {
      float32x4_t sum_vec = vdupq_n_f32(0.0f);
      const float* x_ptr = x.data();
      size_t col = 0;

      const size_t num_blocks_simd = n_cols / block_size;

      for (size_t block_idx = 0; block_idx < num_blocks_simd; ++block_idx) {
        const uint8_t* block_ptr =
            w_data + (row * blocks_per_row + block_idx) * bytes_per_block;
        const uint16_t* f16_scale_ptr =
            reinterpret_cast<const uint16_t*>(block_ptr);
        const uint8_t* quants_ptr = block_ptr + sizeof(uint16_t);

        float scale = f16_to_f32(*f16_scale_ptr);

        const float32x4_t scale_vec = vdupq_n_f32(scale);
        const int8x16_t s8_8 = vdupq_n_s8(8);
        const uint8x16_t q_packed = vld1q_u8(quants_ptr);

        const uint8x16_t q_l = vandq_u8(q_packed, vdupq_n_u8(0x0F));
        const int8x16_t q_l_s8 = vsubq_s8(vreinterpretq_s8_u8(q_l), s8_8);

        const uint8x16_t q_h = vshrq_n_u8(q_packed, 4);
        const int8x16_t q_h_s8 = vsubq_s8(vreinterpretq_s8_u8(q_h), s8_8);

        // dequantize and dot product for low nibbles
        {
          const int16x8_t q_s16_lo = vmovl_s8(vget_low_s8(q_l_s8));
          const int16x8_t q_s16_hi = vmovl_s8(vget_high_s8(q_l_s8));

          const float32x4_t w0 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_lo))), scale_vec);
          const float32x4_t w1 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_lo))), scale_vec);
          const float32x4_t w2 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_hi))), scale_vec);
          const float32x4_t w3 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_hi))), scale_vec);

          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 0), w0);
          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 4), w1);
          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 8), w2);
          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 12), w3);
        }

        // dequantize and dot product for high nibbles
        {
          const int16x8_t q_s16_lo = vmovl_s8(vget_low_s8(q_h_s8));
          const int16x8_t q_s16_hi = vmovl_s8(vget_high_s8(q_h_s8));

          const float32x4_t w0 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_lo))), scale_vec);
          const float32x4_t w1 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_lo))), scale_vec);
          const float32x4_t w2 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_low_s16(q_s16_hi))), scale_vec);
          const float32x4_t w3 = vmulq_f32(
              vcvtq_f32_s32(vmovl_s16(vget_high_s16(q_s16_hi))), scale_vec);

          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 16), w0);
          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 20), w1);
          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 24), w2);
          sum_vec = vfmaq_f32(sum_vec, vld1q_f32(x_ptr + col + 28), w3);
        }
        col += block_size;
      }

      float sum = vaddvq_f32(sum_vec);

      // Scalar remainder for remainder of blocks
      for (size_t block_idx = num_blocks_simd; block_idx < blocks_per_row;
           ++block_idx) {
        const uint8_t* block_ptr =
            w_data + (row * blocks_per_row + block_idx) * bytes_per_block;
        const uint16_t* f16_scale_ptr =
            reinterpret_cast<const uint16_t*>(block_ptr);
        const uint8_t* quants_ptr = block_ptr + sizeof(uint16_t);

        float scale = f16_to_f32(*f16_scale_ptr);

        const int half_block_size = 16;
        for (int i = 0; i < half_block_size; ++i) {
          if (col + i >= n_cols) break;
          uint8_t q_low = quants_ptr[i] & 0x0F;
          float w_val = dequantize_q4_0(q_low, scale);
          sum += w_val * x[col + i];
        }
        for (int i = 0; i < half_block_size; ++i) {
          if (col + half_block_size + i >= n_cols) break;
          uint8_t q_high = quants_ptr[i] >> 4;
          float w_val = dequantize_q4_0(q_high, scale);
          sum += w_val * x[col + half_block_size + i];
        }
        col += block_size;
      }

      o[row] = sum;
    }
#elif defined(__x86_64__)
    // Process each output row
    for (size_t row = start_row; row < end_row; ++row) {
      __m256 sum_vec = _mm256_setzero_ps();
      const float* x_ptr = x.data();
      size_t col = 0;

      const size_t num_blocks_simd = n_cols / block_size;

      for (size_t block_idx = 0; block_idx < num_blocks_simd; ++block_idx) {
        const uint8_t* block_ptr =
            w_data + (row * blocks_per_row + block_idx) * bytes_per_block;
        const uint16_t* f16_scale_ptr =
            reinterpret_cast<const uint16_t*>(block_ptr);
        const uint8_t* quants_ptr = block_ptr + sizeof(uint16_t);

        float scale = f16_to_f32(*f16_scale_ptr);

        const __m256 scale_vec = _mm256_set1_ps(scale);
        const __m128i s8_8 = _mm_set1_epi8(8);
        const __m128i q_packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i*>(quants_ptr));

        const __m128i q_l = _mm_and_si128(q_packed, _mm_set1_epi8(0x0F));
        const __m128i q_l_s8 = _mm_sub_epi8(q_l, s8_8);

        const __m128i q_h =
            _mm_srli_epi16(_mm_and_si128(q_packed, _mm_set1_epi8(0xF0)), 4);
        const __m128i q_h_s8 = _mm_sub_epi8(q_h, s8_8);

        // dequantize and dot product for low nibbles
        {
          const __m128i q_s16_lo = _mm_cvtepi8_epi16(q_l_s8);
          const __m128i q_s16_hi = _mm_cvtepi8_epi16(_mm_srli_si128(q_l_s8, 8));

          const __m256 w0 = _mm256_mul_ps(
              _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(q_s16_lo)), scale_vec);
          const __m256 w1 = _mm256_mul_ps(
              _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(q_s16_hi)), scale_vec);

          sum_vec =
              _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + col + 0), w0, sum_vec);
          sum_vec =
              _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + col + 8), w1, sum_vec);
        }

        // dequantize and dot product for high nibbles
        {
          const __m128i q_s16_lo = _mm_cvtepi8_epi16(q_h_s8);
          const __m128i q_s16_hi = _mm_cvtepi8_epi16(_mm_srli_si128(q_h_s8, 8));

          const __m256 w0 = _mm256_mul_ps(
              _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(q_s16_lo)), scale_vec);
          const __m256 w1 = _mm256_mul_ps(
              _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(q_s16_hi)), scale_vec);

          sum_vec =
              _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + col + 16), w0, sum_vec);
          sum_vec =
              _mm256_fmadd_ps(_mm256_loadu_ps(x_ptr + col + 24), w1, sum_vec);
        }
        col += block_size;
      }
      __m128 sum_vec_128 = _mm_add_ps(_mm256_extractf128_ps(sum_vec, 0),
                                      _mm256_extractf128_ps(sum_vec, 1));
      __m128 sum_vec_shuffled = _mm_hadd_ps(sum_vec_128, sum_vec_128);
      sum_vec_shuffled = _mm_hadd_ps(sum_vec_shuffled, sum_vec_shuffled);
      float sum = _mm_cvtss_f32(sum_vec_shuffled);

      // Scalar remainder for remainder of blocks
      for (size_t block_idx = num_blocks_simd; block_idx < blocks_per_row;
           ++block_idx) {
        const uint8_t* block_ptr =
            w_data + (row * blocks_per_row + block_idx) * bytes_per_block;
        const uint16_t* f16_scale_ptr =
            reinterpret_cast<const uint16_t*>(block_ptr);
        const uint8_t* quants_ptr = block_ptr + sizeof(uint16_t);

        float scale = f16_to_f32(*f16_scale_ptr);

        const int half_block_size = 16;
        for (int i = 0; i < half_block_size; ++i) {
          if (col + i >= n_cols) break;
          uint8_t q_low = quants_ptr[i] & 0x0F;
          float w_val = dequantize_q4_0(q_low, scale);
          sum += w_val * x[col + i];
        }
        for (int i = 0; i < half_block_size; ++i) {
          if (col + half_block_size + i >= n_cols) break;
          uint8_t q_high = quants_ptr[i] >> 4;
          float w_val = dequantize_q4_0(q_high, scale);
          sum += w_val * x[col + half_block_size + i];
        }
        col += block_size;
      }

      o[row] = sum;
    }
#else
    // Process each output row
    for (size_t row = start_row; row < end_row; ++row) {
      float sum = 0.0f;

      size_t col = 0;

      // Process each Q4_0 block in this row
      for (size_t block = 0; block < blocks_per_row; ++block) {
        const uint8_t* block_ptr =
            w_data + (row * blocks_per_row + block) * bytes_per_block;
        const uint16_t* f16_scale_ptr =
            reinterpret_cast<const uint16_t*>(block_ptr);
        const uint8_t* quants_ptr = block_ptr + sizeof(uint16_t);

        float scale = f16_to_f32(*f16_scale_ptr);

        // Process the 32 values in this block
        const int half_block_size = 16;
        for (int i = 0; i < half_block_size; ++i) {
          if (col + i >= n_cols) break;
          uint8_t q_low = quants_ptr[i] & 0x0F;
          float w_val = dequantize_q4_0(q_low, scale);
          sum += w_val * x[col + i];
        }
        for (int i = 0; i < half_block_size; ++i) {
          if (col + half_block_size + i >= n_cols) break;
          uint8_t q_high = quants_ptr[i] >> 4;
          float w_val = dequantize_q4_0(q_high, scale);
          sum += w_val * x[col + half_block_size + i];
        }
        col += block_size;
      }

      o[row] = sum;
    }
#endif
  };

  std::vector<std::future<void>> results;
  size_t num_threads = g_n_threads;
  size_t chunk_size = (n_rows + num_threads - 1) / num_threads;

  for (size_t i = 0; i < num_threads; ++i) {
    size_t start = i * chunk_size;
    size_t end = std::min(start + chunk_size, n_rows);
    if (start >= end) break;
    results.emplace_back(pool->enqueue(compute_range, start, end));
  }

  for (auto&& result : results) result.get();
}
