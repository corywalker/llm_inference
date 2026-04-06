#include "ops.h"

#include <cmath>
#include <vector>

#include "gguf.h"
#include "gtest/gtest.h"
#include "tensor.h"

bool verbose_g = false;

struct OpsInitializer {
  OpsInitializer() { init_ops(1); }
};
static OpsInitializer ops_initializer;

TEST(OpsTest, RmsNorm) {
  std::vector<float> o(4);
  std::vector<float> x = {1.0, 2.0, 3.0, 4.0};
  rms_norm(o, x, 1e-5f);
  float ss = 1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 4.0;
  ss /= 4.0;
  ss += 1e-5f;
  ss = 1.0f / sqrtf(ss);
  for (size_t i = 0; i < o.size(); ++i) {
    EXPECT_NEAR(o[i], ss * x[i], 1e-6);
  }
}

TEST(OpsTest, Softmax) {
  std::vector<float> x = {1.0, 2.0, 3.0, 4.0};
  softmax(x);
  float max_val = 4.0;
  float sum = expf(1.0 - max_val) + expf(2.0 - max_val) + expf(3.0 - max_val) +
              expf(4.0 - max_val);
  EXPECT_NEAR(x[0], expf(1.0 - max_val) / sum, 1e-6);
  EXPECT_NEAR(x[1], expf(2.0 - max_val) / sum, 1e-6);
  EXPECT_NEAR(x[2], expf(3.0 - max_val) / sum, 1e-6);
  EXPECT_NEAR(x[3], expf(4.0 - max_val) / sum, 1e-6);
}

TEST(OpsTest, Rope) {
  tensor_3 t = {{{1.0f, 2.0f, 3.0f, 4.0f}}};
  int n_rot = 4;
  float rope_freq_base = 10000.0f;
  float rope_freq_scale = 1.0f;
  int pos = 0;

  rope(t, n_rot, rope_freq_base, rope_freq_scale, pos);

  EXPECT_NEAR(t[0][0][0], 1.0f, 1e-6);
  EXPECT_NEAR(t[0][0][1], 2.0f, 1e-6);
  EXPECT_NEAR(t[0][0][2], 3.0f, 1e-6);
  EXPECT_NEAR(t[0][0][3], 4.0f, 1e-6);

  t = {{{1.0f, 2.0f, 3.0f, 4.0f}}};
  pos = 1;
  rope(t, n_rot, rope_freq_base, rope_freq_scale, pos);

  EXPECT_NEAR(t[0][0][0], -1.984111f, 1e-4);
  EXPECT_NEAR(t[0][0][2], 2.462337f, 1e-4);
}

TEST(OpsTest, Scale) {
  tensor_3 t = {{{1.0f, 2.0f}, {3.0f, 4.0f}}};
  scale(t, 2.0f);
  EXPECT_NEAR(t[0][0][0], 2.0f, 1e-6);
  EXPECT_NEAR(t[0][0][1], 4.0f, 1e-6);
  EXPECT_NEAR(t[0][1][0], 6.0f, 1e-6);
  EXPECT_NEAR(t[0][1][1], 8.0f, 1e-6);
}

TEST(OpsTest, MatVecMulFP16) {
  size_t n_rows = 2;
  size_t n_cols = 4;
  std::vector<float> o(n_rows);
  std::vector<uint16_t> w(n_rows * n_cols);

  // Weights: {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}
  // 1.0 = 0x3c00, 2.0 = 0x4000, 3.0 = 0x4200, 4.0 = 0x4400
  // 5.0 = 0x4500, 6.0 = 0x4600, 7.0 = 0x4700, 8.0 = 0x4800
  w = {0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800};

  std::vector<float> x = {0.5f, 0.5f, 0.5f, 0.5f};

  // Row 0: 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 0.5(10) = 5.0
  // Row 1: 5*0.5 + 6*0.5 + 7*0.5 + 8*0.5 = 0.5(26) = 13.0

  mat_vec_mul_fp16(o, w, x, n_rows, n_cols);

  EXPECT_NEAR(o[0], 5.0f, 1e-3);
  EXPECT_NEAR(o[1], 13.0f, 1e-3);
}

// Helper to create a minimal GGUF buffer for a single tensor
std::vector<uint8_t> create_minimal_gguf(const std::string& name,
                                         const std::vector<uint64_t>& shape,
                                         GGUFTensorType type,
                                         const std::vector<uint8_t>& data) {
  std::vector<uint8_t> buffer(2048 + data.size());
  size_t offset = 0;

  // Header
  GGUFHeader header = {GGUF_MAGIC, GGUF_VERSION, 1, 0};
  memcpy(buffer.data() + offset, &header, sizeof(header));
  offset += sizeof(header);

  // Tensor info
  uint64_t name_len = name.length();
  memcpy(buffer.data() + offset, &name_len, sizeof(name_len));
  offset += sizeof(name_len);
  memcpy(buffer.data() + offset, name.c_str(), name_len);
  offset += name_len;

  uint32_t dims = shape.size();
  memcpy(buffer.data() + offset, &dims, sizeof(dims));
  offset += sizeof(dims);
  memcpy(buffer.data() + offset, shape.data(), dims * sizeof(uint64_t));
  offset += dims * sizeof(uint64_t);

  uint32_t tensor_type = static_cast<uint32_t>(type);
  memcpy(buffer.data() + offset, &tensor_type, sizeof(tensor_type));
  offset += sizeof(tensor_type);

  uint64_t tensor_offset = 0;
  memcpy(buffer.data() + offset, &tensor_offset, sizeof(tensor_offset));
  offset += sizeof(tensor_offset);

  // Align to 32 bytes for data section
  size_t alignment = 32;
  size_t data_section_start = (offset + alignment - 1) & ~(alignment - 1);
  memcpy(buffer.data() + data_section_start, data.data(), data.size());

  buffer.resize(data_section_start + data.size());
  return buffer;
}

TEST(OpsTest, MatVecMul_Q4_K) {
  const size_t n_cols = 256;
  const size_t n_rows = 1;

  block_q4_K block;
  memset(&block, 0, sizeof(block));
  block.d = f32_to_f16(1.0f);
  block.dmin = f32_to_f16(0.0f);
  // Set scales and mins so that final weights are 2.0.
  // block.scales[0..3] = 1 (ls for 0..3)
  // block.scales[8..11] = 1 (ls for 4..7)
  memset(block.scales, 0, 12);
  for (int i = 0; i < 4; ++i) block.scales[i] = 1;
  for (int i = 8; i < 12; ++i) block.scales[i] = 1;

  // All quants = 2 -> 1.0 * 1 * 2 - 0 = 2.0
  memset(block.qs, 2 | (2 << 4), sizeof(block.qs));

  std::vector<uint8_t> data(sizeof(block));
  memcpy(data.data(), &block, sizeof(block));

  auto gguf_buf =
      create_minimal_gguf("w", {n_cols, n_rows}, GGUFTensorType::Q4_K, data);
  GGUFFile gguf_file(gguf_buf.data(), gguf_buf.size());

  std::vector<float> x(n_cols, 1.0f);
  std::vector<float> o;

  mat_vec_mul_q4_k(o, gguf_file.get_tensor_infos()[0], gguf_file, x);

  ASSERT_EQ(o.size(), 1);
  // 256 elements * 2.0 = 512.0
  EXPECT_NEAR(o[0], 512.0f, 1e-3);
}

TEST(OpsTest, MatVecMul_Q6_K) {
  const size_t n_cols = 256;
  const size_t n_rows = 1;

  block_q6_K block;
  memset(&block, 0, sizeof(block));
  block.d = f32_to_f16(1.0f);
  for (int i = 0; i < 16; ++i) block.scales[i] = 1;
  // Set all to 1.0 weight:
  // q = 1 requires (ql & 0xF) = 1 and (qh & 3) = 2.
  memset(block.ql, 0x11, sizeof(block.ql));
  memset(block.qh, 0xAA,
         sizeof(block.qh));  // 0xAA = 10101010, all 2-bit pairs are 2.

  std::vector<uint8_t> data(sizeof(block));
  memcpy(data.data(), &block, sizeof(block));

  auto gguf_buf =
      create_minimal_gguf("w", {n_cols, n_rows}, GGUFTensorType::Q6_K, data);
  GGUFFile gguf_file(gguf_buf.data(), gguf_buf.size());

  std::vector<float> x(n_cols, 1.0f);
  std::vector<float> o;

  mat_vec_mul_q6_k(o, gguf_file.get_tensor_infos()[0], gguf_file, x);

  ASSERT_EQ(o.size(), 1);
  // 256 elements * 1.0 = 256.0
  EXPECT_NEAR(o[0], 256.0f, 1e-3);
}