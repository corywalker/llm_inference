#include "ops.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
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

static inline int nearest_int(float fval) {
  assert(std::abs(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// Reference: llama.cpp/ggml/src/ggml-quants.c : quantize_row_q8_0
void quantize_row_q8_0(const std::vector<float>& x, std::vector<BlockQ8_0>& y,
                       size_t size) {
  assert(size % 32 == 0);
  size_t nb = size / 32;
  y.resize(nb);

  for (size_t i = 0; i < nb; i++) {
    float amax = 0.0f;
    for (int j = 0; j < 32; j++) {
      const float v = std::abs(x[i * 32 + j]);
      if (amax < v) amax = v;
    }

    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = f32_to_f16(d);

    for (int j = 0; j < 32; j++) {
      const float x0 = x[i * 32 + j] * id;
      y[i].qs[j] = nearest_int(x0);
    }
  }
}

// Quantized matrix-vector multiplication for Q4_0 weights
// Reference: llama.cpp ggml_vec_dot_q4_0_q8_0
// ARM:
// https://github.com/ggml-org/llama.cpp/blob/2aa45ef9e31069ea0a7d0fef7ce858facdf25218/ggml/src/ggml-cpu/arch/arm/quants.c#L140
// x86:
// https://github.com/ggml-org/llama.cpp/blob/ddcb75dd8ac42dc23eb84f13bb17670fe9f2d49b/ggml/src/ggml-cpu/arch/x86/quants.c#L543
// Note that ggml_gemv_q4_0_4x4_q8_0 would be used instead of this if repacking
// is enabled.
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
  const size_t block_size = 32;
  const size_t bytes_per_block = 2 + 16;  // 2 bytes scale + 16 bytes quants
  const size_t blocks_per_row = (n_cols + block_size - 1) / block_size;
  const uint8_t* w_data = gguf_file.get_tensor_data(w_tensor);

  // Pre-quantize x to Q8_0 for ARM DOT optimization
#if defined(__aarch64__) || defined(__arm64__)
  std::vector<BlockQ8_0> x_q8_0;
  quantize_row_q8_0(x, x_q8_0, n_cols);
#endif

  auto compute_range = [&](size_t start_row, size_t end_row) {
#if defined(__aarch64__) || defined(__arm64__)
    for (size_t row = start_row; row < end_row; ++row) {
      float32x4_t sum_v0 = vdupq_n_f32(0.0f);
      float32x4_t sum_v1 = vdupq_n_f32(0.0f);
      float sum_remaining = 0.0f;

      const BlockQ8_0* x_blocks = x_q8_0.data();
      const uint8_t* row_w_ptr =
          w_data + row * blocks_per_row * bytes_per_block;

      const uint8x16_t mask_low = vdupq_n_u8(0x0F);
      const int8x16_t v_8 = vdupq_n_s8(8);

      size_t block_idx = 0;

      // Main loop unrolled 2x
      for (; block_idx + 1 < blocks_per_row; block_idx += 2) {
        const uint8_t* block_ptr_0 = row_w_ptr + block_idx * bytes_per_block;
        const uint8_t* block_ptr_1 =
            row_w_ptr + (block_idx + 1) * bytes_per_block;

        // --- 4-bit -> 8-bit conversion ---
        const uint8_t* w_quants_ptr_0 = block_ptr_0 + sizeof(uint16_t);
        const uint8_t* w_quants_ptr_1 = block_ptr_1 + sizeof(uint16_t);

        uint8x16_t v_q4_0 = vld1q_u8(w_quants_ptr_0);
        uint8x16_t v_q4_1 = vld1q_u8(w_quants_ptr_1);

        int8x16_t v_w_lo_0 =
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v_q4_0, mask_low)), v_8);
        int8x16_t v_w_hi_0 =
            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v_q4_0, 4)), v_8);

        int8x16_t v_w_lo_1 =
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v_q4_1, mask_low)), v_8);
        int8x16_t v_w_hi_1 =
            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v_q4_1, 4)), v_8);

        // --- Load scales ---
        const uint16_t* f16_scale_ptr_0 =
            reinterpret_cast<const uint16_t*>(block_ptr_0);
        const uint16_t* f16_scale_ptr_1 =
            reinterpret_cast<const uint16_t*>(block_ptr_1);

        float w_scale_0 = f16_to_f32(*f16_scale_ptr_0);
        float w_scale_1 = f16_to_f32(*f16_scale_ptr_1);

        float x_scale_0 = f16_to_f32(x_blocks[block_idx].d);
        float x_scale_1 = f16_to_f32(x_blocks[block_idx + 1].d);

        float combined_scale_0 = w_scale_0 * x_scale_0;
        float combined_scale_1 = w_scale_1 * x_scale_1;

        // --- Load x ---
        const int8_t* x_quants_ptr_0 = x_blocks[block_idx].qs;
        const int8_t* x_quants_ptr_1 = x_blocks[block_idx + 1].qs;

        int8x16_t v_x_lo_0 = vld1q_s8(x_quants_ptr_0);
        int8x16_t v_x_hi_0 = vld1q_s8(x_quants_ptr_0 + 16);

        int8x16_t v_x_lo_1 = vld1q_s8(x_quants_ptr_1);
        int8x16_t v_x_hi_1 = vld1q_s8(x_quants_ptr_1 + 16);

        // --- Dot products ---
        int32x4_t dot_0 = vdupq_n_s32(0);
        dot_0 = vdotq_s32(dot_0, v_w_lo_0, v_x_lo_0);
        dot_0 = vdotq_s32(dot_0, v_w_hi_0, v_x_hi_0);

        sum_v0 = vmlaq_n_f32(sum_v0, vcvtq_f32_s32(dot_0), combined_scale_0);

        int32x4_t dot_1 = vdupq_n_s32(0);
        dot_1 = vdotq_s32(dot_1, v_w_lo_1, v_x_lo_1);
        dot_1 = vdotq_s32(dot_1, v_w_hi_1, v_x_hi_1);

        sum_v1 = vmlaq_n_f32(sum_v1, vcvtq_f32_s32(dot_1), combined_scale_1);
      }

      // Remainder loop
      for (; block_idx < blocks_per_row; ++block_idx) {
        const uint8_t* block_ptr = row_w_ptr + block_idx * bytes_per_block;

        const uint16_t* f16_scale_ptr =
            reinterpret_cast<const uint16_t*>(block_ptr);
        float w_scale = f16_to_f32(*f16_scale_ptr);
        float x_scale = f16_to_f32(x_blocks[block_idx].d);
        float combined_scale = w_scale * x_scale;

        const uint8_t* w_quants_ptr = block_ptr + sizeof(uint16_t);
        uint8x16_t v_q4 = vld1q_u8(w_quants_ptr);
        int8x16_t v_w_lo =
            vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v_q4, mask_low)), v_8);
        int8x16_t v_w_hi =
            vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v_q4, 4)), v_8);

        const int8_t* x_quants_ptr = x_blocks[block_idx].qs;
        int8x16_t v_x_lo = vld1q_s8(x_quants_ptr);
        int8x16_t v_x_hi = vld1q_s8(x_quants_ptr + 16);

        int32x4_t dot = vdupq_n_s32(0);
        dot = vdotq_s32(dot, v_w_lo, v_x_lo);
        dot = vdotq_s32(dot, v_w_hi, v_x_hi);

        int32_t scalar_dot = vaddvq_s32(dot);
        sum_remaining += scalar_dot * combined_scale;
      }

      o[row] = vaddvq_f32(sum_v0) + vaddvq_f32(sum_v1) + sum_remaining;
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

// llama.cpp analogue would be ggml_vec_dot_f16 here.
// FP16 Matrix-Vector Multiplication
void mat_vec_mul_fp16(std::vector<float>& o, const std::vector<uint16_t>& w,
                      const std::vector<float>& x, size_t n_rows,
                      size_t n_cols) {
  if (x.size() != n_cols) {
    throw std::runtime_error("mat_vec_mul_fp16: input vector size mismatch");
  }
  if (w.size() != n_rows * n_cols) {
    throw std::runtime_error("mat_vec_mul_fp16: weight matrix size mismatch");
  }
  o.resize(n_rows);

#if defined(__aarch64__) || defined(__arm64__)
  // 1. Convert input x (F32) to x_f16 (F16) using NEON
  // We align to 8 elements (128-bit for FP16 is 8x16-bit)
  std::vector<float16_t> x_f16(n_cols);
  size_t j = 0;
  for (; j + 3 < n_cols; j += 4) {
    float32x4_t f32_val = vld1q_f32(x.data() + j);
    float16x4_t f16_val = vcvt_f16_f32(f32_val);
    vst1_f16(reinterpret_cast<float16_t*>(x_f16.data() + j), f16_val);
  }
  for (; j < n_cols; ++j) {
    x_f16[j] = static_cast<float16_t>(x[j]);
  }

  auto compute_range = [&](size_t start_row, size_t end_row) {
    for (size_t i = start_row; i < end_row; ++i) {
      const float16_t* w_ptr =
          reinterpret_cast<const float16_t*>(w.data() + i * n_cols);
      const float16_t* x_ptr = x_f16.data();

      // Accumulators
      float16x8_t sum_vec0 = vdupq_n_f16(0.0f);
      float16x8_t sum_vec1 = vdupq_n_f16(0.0f);
      float16x8_t sum_vec2 = vdupq_n_f16(0.0f);
      float16x8_t sum_vec3 = vdupq_n_f16(0.0f);

      size_t k = 0;
      // Unroll 4x (32 elements per loop)
      for (; k + 31 < n_cols; k += 32) {
        float16x8_t w0 = vld1q_f16(w_ptr + k);
        float16x8_t x0 = vld1q_f16(x_ptr + k);
        sum_vec0 = vfmaq_f16(sum_vec0, w0, x0);

        float16x8_t w1 = vld1q_f16(w_ptr + k + 8);
        float16x8_t x1 = vld1q_f16(x_ptr + k + 8);
        sum_vec1 = vfmaq_f16(sum_vec1, w1, x1);

        float16x8_t w2 = vld1q_f16(w_ptr + k + 16);
        float16x8_t x2 = vld1q_f16(x_ptr + k + 16);
        sum_vec2 = vfmaq_f16(sum_vec2, w2, x2);

        float16x8_t w3 = vld1q_f16(w_ptr + k + 24);
        float16x8_t x3 = vld1q_f16(x_ptr + k + 24);
        sum_vec3 = vfmaq_f16(sum_vec3, w3, x3);
      }

      // Reduce accumulators
      sum_vec0 = vaddq_f16(sum_vec0, sum_vec1);
      sum_vec2 = vaddq_f16(sum_vec2, sum_vec3);
      sum_vec0 = vaddq_f16(sum_vec0, sum_vec2);

      // Handle remaining blocks of 8
      for (; k + 7 < n_cols; k += 8) {
        float16x8_t w_val = vld1q_f16(w_ptr + k);
        float16x8_t x_val = vld1q_f16(x_ptr + k);
        sum_vec0 = vfmaq_f16(sum_vec0, w_val, x_val);
      }

      float result = vaddvq_f32(vcvt_f32_f16(
          vadd_f16(vget_low_f16(sum_vec0), vget_high_f16(sum_vec0))));

      // Handle remainders (scalar)
      for (; k < n_cols; ++k) {
        result += (float)w_ptr[k] * (float)x_ptr[k];
      }
      o[i] = result;
    }
  };
#else
  auto compute_range = [&](size_t start_row, size_t end_row) {
    for (size_t i = start_row; i < end_row; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < n_cols; ++j) {
        float w_val = f16_to_f32(w[i * n_cols + j]);
        sum += w_val * x[j];
      }
      o[i] = sum;
    }
  };
#endif

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

void vec_scale_f16(tensor_f16_1& y, float v) {
  for (size_t i = 0; i < y.size(); ++i) {
    float val = f16_to_f32(y[i]);
    y[i] = f32_to_f16(val * v);
  }
}

void vec_mad_f16(tensor_f16_1& y, const tensor_f16_1& x, float v) {
  // Assuming y and x have same size
  size_t n = y.size();
  for (size_t i = 0; i < n; ++i) {
    float y_val = f16_to_f32(y[i]);
    float x_val = f16_to_f32(x[i]);
    y[i] = f32_to_f16(y_val + x_val * v);
  }
}
