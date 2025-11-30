#include "gguf.h"

#include <cstdio>
#include <fstream>

#include "gtest/gtest.h"
#include "ops.h"

bool verbose_g = false;

struct OpsInitializer {
  OpsInitializer() { init_ops(1); }
};
static OpsInitializer ops_initializer;

// Helper function to write a string in GGUF format
void write_string(std::ofstream& file, const std::string& s) {
  uint64_t len = s.length();
  file.write(reinterpret_cast<const char*>(&len), sizeof(len));
  file.write(s.c_str(), len);
}

// Helper function to create a test GGUF file
void create_test_gguf_file(const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);

  // Header
  GGUFHeader header = {GGUF_MAGIC, GGUF_VERSION, 1, 1};
  file.write(reinterpret_cast<const char*>(&header), sizeof(header));

  // Metadata
  write_string(file, "test_key");
  uint32_t type = static_cast<uint32_t>(GGUFType::STRING);
  file.write(reinterpret_cast<const char*>(&type), sizeof(type));
  write_string(file, "test_value");

  // Tensor Info
  write_string(file, "test_tensor");
  uint32_t dims = 1;
  file.write(reinterpret_cast<const char*>(&dims), sizeof(dims));
  uint64_t shape = 4;
  file.write(reinterpret_cast<const char*>(&shape), sizeof(shape));
  uint32_t tensor_type = static_cast<uint32_t>(GGUFTensorType::F32);
  file.write(reinterpret_cast<const char*>(&tensor_type), sizeof(tensor_type));
  uint64_t offset = 0;
  file.write(reinterpret_cast<const char*>(&offset), sizeof(offset));

  // Padding
  size_t current_pos = file.tellp();
  size_t alignment = 32;
  size_t padding = (alignment - (current_pos % alignment)) % alignment;
  for (size_t i = 0; i < padding; ++i) {
    file.put(0);
  }

  // Tensor Data
  float tensor_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  file.write(reinterpret_cast<const char*>(tensor_data), sizeof(tensor_data));

  file.close();
}

TEST(GGUFTest, F16toF32) {
  // Test cases for f16_to_f32
  // 0 -> 0
  EXPECT_FLOAT_EQ(f16_to_f32(0x0000), 0.0f);
  // 1 -> 1
  EXPECT_FLOAT_EQ(f16_to_f32(0x3C00), 1.0f);
  // -1 -> -1
  EXPECT_FLOAT_EQ(f16_to_f32(0xBC00), -1.0f);
  // 0.5 -> 0.5
  EXPECT_FLOAT_EQ(f16_to_f32(0x3800), 0.5f);
  // 65504 -> 65504 (max value)
  EXPECT_FLOAT_EQ(f16_to_f32(0x7BFF), 65504.0f);

  ASSERT_NEAR(f16_to_f32(7426), 0.004890, 1e-6);
  ASSERT_NEAR(f16_to_f32(40759), -0.007046, 1e-6);
  // We know it should be this because
  // gguf-tools inspect-tensor
  // ~/gemma-3-1b-it-qat-q4_0-gguf/gemma-3-1b-it-q4_0.gguf token_embd.weight  |
  // less
  ASSERT_NEAR(f16_to_f32(773), 0.000046, 1e-6);
}

TEST(GGUFTest, DequantizeQ4_0) {
  // Test cases for dequantize_q4_0
  // q4 = 0, d = 1.0 -> -8.0
  EXPECT_FLOAT_EQ(dequantize_q4_0(0, 1.0), -8.0f);
  // q4 = 8, d = 1.0 -> 0.0
  EXPECT_FLOAT_EQ(dequantize_q4_0(8, 1.0), 0.0f);
  // q4 = 15, d = 1.0 -> 7.0
  EXPECT_FLOAT_EQ(dequantize_q4_0(15, 1.0), 7.0f);
  // q4 = 0, d = 0.5 -> -4.0
  EXPECT_FLOAT_EQ(dequantize_q4_0(0, 0.5), -4.0f);
}

TEST(GGUFTest, TensorTypeToString) {
  EXPECT_EQ(tensorTypeToString(static_cast<uint32_t>(GGUFTensorType::F32)),
            "F32");
  EXPECT_EQ(tensorTypeToString(static_cast<uint32_t>(GGUFTensorType::F16)),
            "F16");
  EXPECT_EQ(tensorTypeToString(static_cast<uint32_t>(GGUFTensorType::Q4_0)),
            "Q4_0");
  EXPECT_EQ(tensorTypeToString(12345), "UNKNOWN");
}

TEST(GGUFTest, LoadFile) {
  const std::string filename = "/tmp/test.gguf";
  create_test_gguf_file(filename);

  GGUFFile gguf_file(filename);

  // Check header
  const GGUFHeader& header = gguf_file.get_header();
  EXPECT_EQ(header.magic, GGUF_MAGIC);
  EXPECT_EQ(header.version, GGUF_VERSION);
  EXPECT_EQ(header.tensor_count, 1);
  EXPECT_EQ(header.metadata_kv_count, 1);

  // Check metadata
  const GGUFValueMap& metadata = gguf_file.get_metadata();
  EXPECT_EQ(metadata.size(), 1);
  EXPECT_EQ(metadata.at("test_key").type, GGUFType::STRING);
  EXPECT_EQ(metadata.at("test_key").str, "test_value");

  // Check tensor info
  const std::vector<TensorInfo>& tensor_infos = gguf_file.get_tensor_infos();
  EXPECT_EQ(tensor_infos.size(), 1);
  const TensorInfo& tensor_info = tensor_infos[0];
  EXPECT_EQ(tensor_info.name, "test_tensor");
  EXPECT_EQ(tensor_info.shape.size(), 1);
  EXPECT_EQ(tensor_info.shape[0], 4);
  EXPECT_EQ(tensor_info.tensor_type,
            static_cast<uint32_t>(GGUFTensorType::F32));
  EXPECT_EQ(tensor_info.tensor_offset, 0);

  // Check tensor data
  std::vector<float> tensor_data(4);
  gguf_file.read_tensor_data(tensor_info, tensor_data.data(),
                             tensor_data.size() * sizeof(float));
  EXPECT_FLOAT_EQ(tensor_data[0], 1.0f);
  EXPECT_FLOAT_EQ(tensor_data[1], 2.0f);
  EXPECT_FLOAT_EQ(tensor_data[2], 3.0f);
  EXPECT_FLOAT_EQ(tensor_data[3], 4.0f);

  // Clean up the test file
  std::remove(filename.c_str());
}

TEST(GGUFTest, Q4_0_MatVecMul) {
  // Create a simple Q4_0 tensor in memory and test mat_vec_mul_q4_0
  // against dequantize + regular mat_vec_mul

  // Create a small Q4_0 weight matrix: 4 rows x 8 cols
  // This requires 8/32 = 0.25 blocks per row, so we need 1 block per row (32
  // elements, but only 8 used)
  const size_t n_rows = 4;
  const size_t n_cols = 8;

  // Create Q4_0 data: 1 block per row
  // Each block: 2 bytes scale + 16 bytes quants (32 values)
  std::vector<uint8_t> q4_data;

  // Use different patterns for each row
  uint8_t patterns[] = {0xF0, 0xE1, 0xD2, 0xC3};
  uint16_t f16_scales[] = {0x3800, 0x3666, 0x3333,
                           0x3000};  // 0.5, 0.4, 0.3, 0.2 in f16 approx

  for (size_t row = 0; row < n_rows; row++) {
    uint16_t scale = f16_scales[row];
    q4_data.push_back(scale & 0xFF);
    q4_data.push_back((scale >> 8) & 0xFF);
    for (int i = 0; i < 16; i++) {
      q4_data.push_back(patterns[row]);
    }
  }

  // Create a minimal GGUF file in memory
  std::vector<uint8_t> gguf_buffer(1024);
  size_t offset = 0;

  // Header
  GGUFHeader header = {GGUF_MAGIC, GGUF_VERSION, 1, 0};
  memcpy(gguf_buffer.data() + offset, &header, sizeof(header));
  offset += sizeof(header);

  // Tensor info
  std::string tensor_name = "test_weight";
  uint64_t name_len = tensor_name.length();
  memcpy(gguf_buffer.data() + offset, &name_len, sizeof(name_len));
  offset += sizeof(name_len);
  memcpy(gguf_buffer.data() + offset, tensor_name.c_str(), name_len);
  offset += name_len;

  uint32_t dims = 2;
  memcpy(gguf_buffer.data() + offset, &dims, sizeof(dims));
  offset += sizeof(dims);

  uint64_t shape[2] = {n_cols, n_rows};
  memcpy(gguf_buffer.data() + offset, shape, sizeof(shape));
  offset += sizeof(shape);

  uint32_t tensor_type = static_cast<uint32_t>(GGUFTensorType::Q4_0);
  memcpy(gguf_buffer.data() + offset, &tensor_type, sizeof(tensor_type));
  offset += sizeof(tensor_type);

  uint64_t tensor_offset = 0;
  memcpy(gguf_buffer.data() + offset, &tensor_offset, sizeof(tensor_offset));
  offset += sizeof(tensor_offset);

  // Align to 32 bytes for data section
  size_t alignment = 32;
  size_t data_section_start = (offset + alignment - 1) & ~(alignment - 1);

  // Copy Q4_0 data
  memcpy(gguf_buffer.data() + data_section_start, q4_data.data(),
         q4_data.size());

  gguf_buffer.resize(data_section_start + q4_data.size());

  // Load GGUF file
  GGUFFile gguf_file(gguf_buffer.data(), gguf_buffer.size());
  const auto& tensor_infos = gguf_file.get_tensor_infos();
  ASSERT_EQ(tensor_infos.size(), 1);
  const TensorInfo& tensor = tensor_infos[0];

  // Create input vector
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  ASSERT_EQ(input.size(), n_cols);

  // Method 1: Use mat_vec_mul_q4_0
  std::vector<float> output_q4;
  mat_vec_mul_q4_0(output_q4, tensor, gguf_file, input);

  // Method 2: Manual dequantization and mat_vec_mul
  std::vector<float> output_deq(n_rows);
  for (size_t i = 0; i < n_rows; i++) {
    output_deq[i] = 0.0f;
    float scale = f16_to_f32(f16_scales[i]);
    uint8_t pattern = patterns[i];
    uint8_t q_low = pattern & 0x0F;

    for (size_t j = 0; j < n_cols; j++) {
      float w_val = dequantize_q4_0(q_low, scale);
      output_deq[i] += w_val * input[j];
    }
  }

  // Compare results
  ASSERT_EQ(output_q4.size(), output_deq.size());
  for (size_t i = 0; i < output_q4.size(); i++) {
    EXPECT_NEAR(output_q4[i], output_deq[i], 0.01f)
        << "Mismatch at index " << i << ": Q4 matmul=" << output_q4[i]
        << ", Dequant matmul=" << output_deq[i];
  }
}
