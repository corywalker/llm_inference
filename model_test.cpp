#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "gguf.h"
#include "gtest/gtest.h"
#include "ops.h"

bool verbose_g = false;

struct OpsInitializer {
  OpsInitializer() { init_ops(1); }
};
static OpsInitializer ops_initializer;

struct TensorInfoWriter {
  std::string name;
  GGUFTensorType type;
  std::vector<uint64_t> shape;
  const void* data;
  size_t data_size;
  size_t offset_pos;
};

size_t write_tensor_info(std::vector<uint8_t>& buffer, size_t offset,
                         TensorInfoWriter& writer) {
  // Tensor Info
  uint64_t name_len = writer.name.length();
  memcpy(buffer.data() + offset, &name_len, sizeof(name_len));
  offset += sizeof(name_len);
  memcpy(buffer.data() + offset, writer.name.c_str(), name_len);
  offset += name_len;

  uint32_t dimensions = writer.shape.size();
  memcpy(buffer.data() + offset, &dimensions, sizeof(dimensions));
  offset += sizeof(dimensions);

  memcpy(buffer.data() + offset, writer.shape.data(),
         writer.shape.size() * sizeof(uint64_t));
  offset += writer.shape.size() * sizeof(uint64_t);

  uint32_t tensor_type = (uint32_t)writer.type;
  memcpy(buffer.data() + offset, &tensor_type, sizeof(tensor_type));
  offset += sizeof(tensor_type);

  writer.offset_pos = offset;
  uint64_t tensor_offset = 0;  // Dummy offset
  memcpy(buffer.data() + offset, &tensor_offset, sizeof(tensor_offset));
  offset += sizeof(tensor_offset);

  return offset;
}

// Convert float to IEEE-754 binary16 (float16) bit pattern (uint16_t)
static uint16_t float_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, sizeof(f));
  uint32_t sign = (x >> 31) & 0x1;
  int32_t exp = ((x >> 23) & 0xFF) - 127;
  uint32_t mant = x & 0x7FFFFF;

  if (exp > 15) {
    // overflow -> inf
    return (uint16_t)((sign << 15) | (0x1F << 10));
  } else if (exp <= -15) {
    // underflow -> zero (no subnormals)
    return (uint16_t)(sign << 15);
  } else {
    uint16_t exp16 = (uint16_t)(exp + 15) & 0x1F;
    uint16_t mant16 = (uint16_t)(mant >> 13);
    return (uint16_t)((sign << 15) | (exp16 << 10) | mant16);
  }
}

std::vector<uint8_t> create_q4_0_data(
    size_t num_elements, std::mt19937& rng,
    std::uniform_real_distribution<float>& dist) {
  // Q4_0: blocks of 32 elements. Each block stores a float16 scale (2 bytes)
  // followed by 16 bytes packing 32 4-bit quantized values (nibbles).
  size_t num_blocks = (num_elements + 31) / 32;
  std::vector<uint8_t> data(num_blocks * (2 + 16));
  for (size_t b = 0; b < num_blocks; ++b) {
    // generate up to 32 floats in [-1,1]
    size_t block_elems = std::min<size_t>(32, num_elements - b * 32);
    std::vector<float> vals(32, 0.0f);
    float max_abs = 0.0f;
    for (size_t i = 0; i < block_elems; ++i) {
      vals[i] = dist(rng);
      max_abs = std::max(max_abs, std::fabs(vals[i]));
    }
    if (max_abs < 1e-8f) max_abs = 1e-8f;
    // quantize to signed 4-bit (-8..7). Use scale = max_abs / 7
    float scale = max_abs / 7.0f;
    uint16_t scale_f16 = float_to_f16(scale);
    // place scale
    memcpy(data.data() + b * 18, &scale_f16, sizeof(scale_f16));
    // compute quantized values and pack into 16 bytes
    for (size_t i = 0; i < 16; ++i) {
      int idx0 = 2 * i;
      int idx1 = 2 * i + 1;
      int q0 = 0, q1 = 0;
      if (idx0 < (int)block_elems) {
        q0 = (int)std::lround(vals[idx0] / scale);
      }
      if (idx1 < (int)block_elems) {
        q1 = (int)std::lround(vals[idx1] / scale);
      }
      q0 = std::max(-8, std::min(7, q0));
      q1 = std::max(-8, std::min(7, q1));
      uint8_t nib0 = (uint8_t)(q0 + 8) & 0xF;
      uint8_t nib1 = (uint8_t)(q1 + 8) & 0xF;
      uint8_t packed = (uint8_t)((nib1 << 4) | nib0);
      data[b * 18 + 2 + i] = packed;
    }
  }
  return data;
}

std::vector<uint8_t> create_test_gguf() {
  const uint32_t block_count = 1;
  const uint32_t embedding_length = 32;
  const uint32_t feed_forward_length = 64;
  const uint32_t attention_head_count = 2;
  const uint32_t attention_head_count_kv = 1;
  const uint32_t vocab_size = 10;

  std::vector<uint8_t> buffer(1024 * 1024);  // 1MB should be enough
  size_t offset = 0;

  // Use fixed seed RNG
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Header
  GGUFHeader header = {GGUF_MAGIC, GGUF_VERSION, 13,
                       0};  // metadata_kv_count is set below

  // Metadata
  const char* uint32_keys[] = {"gemma3.block_count", "gemma3.embedding_length",
                               "gemma3.feed_forward_length",
                               "gemma3.attention.head_count",
                               "gemma3.attention.head_count_kv"};
  uint32_t uint32_values[] = {block_count, embedding_length,
                              feed_forward_length, attention_head_count,
                              attention_head_count_kv};
  const char* float_keys[] = {"gemma3.attention.layer_norm_rms_epsilon",
                              "gemma3.rope.freq_base",
                              "gemma3.rope.scaling.factor"};
  float float_values[] = {1e-6f, 1000000.0f, 1.0f};

  header.metadata_kv_count = std::size(uint32_keys) + std::size(float_keys);

  memcpy(buffer.data() + offset, &header, sizeof(header));
  offset += sizeof(header);

  for (size_t i = 0; i < std::size(uint32_keys); ++i) {
    uint64_t key_len = strlen(uint32_keys[i]);
    memcpy(buffer.data() + offset, &key_len, sizeof(key_len));
    offset += sizeof(key_len);
    memcpy(buffer.data() + offset, uint32_keys[i], key_len);
    offset += key_len;

    uint32_t type = (uint32_t)GGUFType::UINT32;
    memcpy(buffer.data() + offset, &type, sizeof(type));
    offset += sizeof(type);

    memcpy(buffer.data() + offset, &uint32_values[i], sizeof(uint32_values[i]));
    offset += sizeof(uint32_values[i]);
  }

  for (size_t i = 0; i < std::size(float_keys); ++i) {
    uint64_t key_len = strlen(float_keys[i]);
    memcpy(buffer.data() + offset, &key_len, sizeof(key_len));
    offset += sizeof(key_len);
    memcpy(buffer.data() + offset, float_keys[i], key_len);
    offset += key_len;

    uint32_t type = (uint32_t)GGUFType::FLOAT32;
    memcpy(buffer.data() + offset, &type, sizeof(type));
    offset += sizeof(type);

    memcpy(buffer.data() + offset, &float_values[i], sizeof(float_values[i]));
    offset += sizeof(float_values[i]);
  }

  // Tokenizer metadata
  std::vector<std::string> tokens = {"<pad>",    "<eos>", "<bos>",    "<unk>",
                                     "▁",        "▁The",  "▁capital", "▁of",
                                     "▁Germany", "▁is",   ":"};
  uint64_t tokens_key_len = strlen("tokenizer.ggml.tokens");
  memcpy(buffer.data() + offset, &tokens_key_len, sizeof(tokens_key_len));
  offset += sizeof(tokens_key_len);
  memcpy(buffer.data() + offset, "tokenizer.ggml.tokens", tokens_key_len);
  offset += tokens_key_len;
  uint32_t array_type = (uint32_t)GGUFType::ARRAY;
  memcpy(buffer.data() + offset, &array_type, sizeof(array_type));
  offset += sizeof(array_type);
  uint32_t element_type = (uint32_t)GGUFType::STRING;
  memcpy(buffer.data() + offset, &element_type, sizeof(element_type));
  offset += sizeof(element_type);
  uint64_t tokens_array_len = tokens.size();
  memcpy(buffer.data() + offset, &tokens_array_len, sizeof(tokens_array_len));
  offset += sizeof(tokens_array_len);
  for (const auto& token : tokens) {
    uint64_t token_len = token.length();
    memcpy(buffer.data() + offset, &token_len, sizeof(token_len));
    offset += sizeof(token_len);
    memcpy(buffer.data() + offset, token.c_str(), token_len);
    offset += token_len;
  }
  header.metadata_kv_count += 1;

  // bos_token_id
  uint64_t bos_key_len = strlen("tokenizer.ggml.bos_token_id");
  memcpy(buffer.data() + offset, &bos_key_len, sizeof(bos_key_len));
  offset += sizeof(bos_key_len);
  memcpy(buffer.data() + offset, "tokenizer.ggml.bos_token_id", bos_key_len);
  offset += bos_key_len;
  uint32_t bos_type = (uint32_t)GGUFType::UINT32;
  memcpy(buffer.data() + offset, &bos_type, sizeof(bos_type));
  offset += sizeof(bos_type);
  uint32_t bos_value = 2;
  memcpy(buffer.data() + offset, &bos_value, sizeof(bos_value));
  offset += sizeof(bos_value);
  header.metadata_kv_count += 1;

  // unk_token_id
  uint64_t unk_key_len = strlen("tokenizer.ggml.unknown_token_id");
  memcpy(buffer.data() + offset, &unk_key_len, sizeof(unk_key_len));
  offset += sizeof(unk_key_len);
  memcpy(buffer.data() + offset, "tokenizer.ggml.unknown_token_id",
         unk_key_len);
  offset += unk_key_len;
  uint32_t unk_type = (uint32_t)GGUFType::UINT32;
  memcpy(buffer.data() + offset, &unk_type, sizeof(unk_type));
  offset += sizeof(unk_type);
  uint32_t unk_value = 3;
  memcpy(buffer.data() + offset, &unk_value, sizeof(unk_value));
  offset += sizeof(unk_value);
  header.metadata_kv_count += 1;

  // Update header with correct metadata_kv_count after adding tokenizer keys
  memcpy(buffer.data(), &header, sizeof(header));

  std::vector<TensorInfoWriter> tensor_writers;

  // token_embd: f16 values randomized in [-1,1]
  std::vector<uint16_t> token_embd_data(embedding_length * vocab_size);
  for (size_t i = 0; i < token_embd_data.size(); ++i) {
    float v = dist(rng);
    token_embd_data[i] = float_to_f16(v);
  }
  tensor_writers.push_back({"token_embd.weight",
                            GGUFTensorType::F16,
                            {embedding_length, vocab_size},
                            token_embd_data.data(),
                            token_embd_data.size() * sizeof(uint16_t)});

  // output_norm: f32
  std::vector<float> output_norm_data(embedding_length);
  for (auto& v : output_norm_data) v = dist(rng);
  tensor_writers.push_back({"output_norm.weight",
                            GGUFTensorType::F32,
                            {embedding_length},
                            output_norm_data.data(),
                            output_norm_data.size() * sizeof(float)});

  std::vector<float> attn_norm_data(embedding_length);
  for (auto& v : attn_norm_data) v = dist(rng);
  tensor_writers.push_back({"blk.0.attn_norm.weight",
                            GGUFTensorType::F32,
                            {embedding_length},
                            attn_norm_data.data(),
                            attn_norm_data.size() * sizeof(float)});

  std::vector<float> attn_q_norm_data(embedding_length);
  for (auto& v : attn_q_norm_data) v = dist(rng);
  tensor_writers.push_back({"blk.0.attn_q_norm.weight",
                            GGUFTensorType::F32,
                            {embedding_length},
                            attn_q_norm_data.data(),
                            attn_q_norm_data.size() * sizeof(float)});

  std::vector<float> attn_k_norm_data(embedding_length);
  for (auto& v : attn_k_norm_data) v = dist(rng);
  tensor_writers.push_back({"blk.0.attn_k_norm.weight",
                            GGUFTensorType::F32,
                            {embedding_length},
                            attn_k_norm_data.data(),
                            attn_k_norm_data.size() * sizeof(float)});

  // Q4 weights for attention
  std::vector<uint8_t> q4_data =
      create_q4_0_data(embedding_length * embedding_length, rng, dist);
  tensor_writers.push_back({"blk.0.attn_q.weight",
                            GGUFTensorType::Q4_0,
                            {embedding_length, embedding_length},
                            q4_data.data(),
                            q4_data.size()});
  tensor_writers.push_back({"blk.0.attn_k.weight",
                            GGUFTensorType::Q4_0,
                            {embedding_length, embedding_length},
                            q4_data.data(),
                            q4_data.size()});
  tensor_writers.push_back({"blk.0.attn_v.weight",
                            GGUFTensorType::Q4_0,
                            {embedding_length, embedding_length},
                            q4_data.data(),
                            q4_data.size()});
  tensor_writers.push_back({"blk.0.attn_output.weight",
                            GGUFTensorType::Q4_0,
                            {embedding_length, embedding_length},
                            q4_data.data(),
                            q4_data.size()});

  std::vector<float> ffn_norm_data(embedding_length);
  for (auto& v : ffn_norm_data) v = dist(rng);
  tensor_writers.push_back({"blk.0.ffn_norm.weight",
                            GGUFTensorType::F32,
                            {embedding_length},
                            ffn_norm_data.data(),
                            ffn_norm_data.size() * sizeof(float)});

  // Q4 ffn weights
  std::vector<uint8_t> q4_ffn_data =
      create_q4_0_data(embedding_length * feed_forward_length, rng, dist);
  tensor_writers.push_back({"blk.0.ffn_gate.weight",
                            GGUFTensorType::Q4_0,
                            {embedding_length, feed_forward_length},
                            q4_ffn_data.data(),
                            q4_ffn_data.size()});
  tensor_writers.push_back({"blk.0.ffn_up.weight",
                            GGUFTensorType::Q4_0,
                            {embedding_length, feed_forward_length},
                            q4_ffn_data.data(),
                            q4_ffn_data.size()});

  std::vector<uint8_t> q4_ffn_down_data =
      create_q4_0_data(feed_forward_length * embedding_length, rng, dist);
  tensor_writers.push_back({"blk.0.ffn_down.weight",
                            GGUFTensorType::Q4_0,
                            {feed_forward_length, embedding_length},
                            q4_ffn_down_data.data(),
                            q4_ffn_down_data.size()});

  for (auto& writer : tensor_writers) {
    offset = write_tensor_info(buffer, offset, writer);
  }

  size_t alignment = 32;
  size_t data_section_start = (offset + alignment - 1) & ~(alignment - 1);
  size_t data_offset = 0;

  for (auto& writer : tensor_writers) {
    uint64_t tensor_offset = data_offset;
    memcpy(buffer.data() + writer.offset_pos, &tensor_offset,
           sizeof(tensor_offset));
    memcpy(buffer.data() + data_section_start + data_offset, writer.data,
           writer.data_size);
    data_offset += writer.data_size;
  }

  buffer.resize(data_section_start + data_offset);
  return buffer;
}

TEST(ModelTest, LoadFromMemory) {
  std::vector<uint8_t> gguf_data = create_test_gguf();
  GGUFFile gguf_file(gguf_data.data(), gguf_data.size());
  Model model(gguf_file);

  ASSERT_EQ(model.hparams().block_count, 1);
  ASSERT_EQ(model.hparams().embedding_length, 32);
  ASSERT_EQ(model.hparams().feed_forward_length, 64);
  ASSERT_EQ(model.hparams().attention_head_count, 2);
  ASSERT_EQ(model.hparams().attention_head_count_kv, 1);
  ASSERT_FLOAT_EQ(model.hparams().f_norm_rms_eps, 1e-6f);

  ASSERT_NE(model.token_embd_weight(), nullptr);
  EXPECT_STREQ(model.token_embd_weight()->name.c_str(), "token_embd.weight");
}

TEST(ModelTest, ForwardPass) {
  std::vector<uint8_t> gguf_data = create_test_gguf();
  GGUFFile gguf_file(gguf_data.data(), gguf_data.size());
  Model model(gguf_file);

  // First forward pass
  std::vector<int> tokens = {1};
  auto result = model.forward(tokens, 0);

  ASSERT_EQ(result.size(), 1);
  const uint32_t vocab_size = 10;
  ASSERT_EQ(result[0].size(), vocab_size);

  const float tolerance = 0.003f;

  // Some tests of approximate values. This will need to update if any random
  // seed or model details change.
  EXPECT_NEAR(result[0][0], 2.9909527f, tolerance);
  EXPECT_NEAR(result[0][1], -0.216222f, tolerance);
  EXPECT_NEAR(result[0][8], 1.6922607f, tolerance);
  EXPECT_NEAR(result[0][9], -2.588623f, tolerance);
  float sum = 0.0f;
  for (auto v : result[0]) sum += v;
  EXPECT_NEAR(sum, 5.2634663581848145f, tolerance);

  // Second forward pass for next token (using KV cache)
  // Find token with largest logit from first pass
  int next_token = 0;
  float max_logit = result[0][0];
  for (size_t i = 1; i < result[0].size(); ++i) {
    if (result[0][i] > max_logit) {
      max_logit = result[0][i];
      next_token = i;
    }
  }

  std::vector<int> tokens2 = {next_token};
  auto result2 = model.forward(tokens2, 1);

  ASSERT_EQ(result2.size(), 1);
  ASSERT_EQ(result2[0].size(), vocab_size);

  // Some tests of approximate values for second token. This will need to update
  // if any random seed or model details change.
  EXPECT_NEAR(result2[0][0], 0.6870570f, tolerance);
  EXPECT_NEAR(result2[0][1], -2.670202f, tolerance);
  EXPECT_NEAR(result2[0][8], 0.1438203f, tolerance);
  EXPECT_NEAR(result2[0][9], -0.409215f, tolerance);
  float sum2 = 0.0f;
  for (auto v : result2[0]) sum2 += v;
  EXPECT_NEAR(sum2, 2.540453f, tolerance);
}

TEST(ModelTest, TokenizeTest) {
  std::vector<uint8_t> gguf_data = create_test_gguf();
  GGUFFile gguf_file(gguf_data.data(), gguf_data.size());
  Model model(gguf_file);

  std::string prompt = "The capital of Germany is:";
  std::vector<int> expected_tokens = {
      2, 5, 6, 7, 8, 9, 10};  // 2 for BOS, then the mapped tokens

  std::vector<int> tokens = model.tokenize(prompt, false);

  ASSERT_EQ(tokens.size(), expected_tokens.size());
  for (size_t i = 0; i < tokens.size(); ++i) {
    ASSERT_EQ(tokens[i], expected_tokens[i]) << "Mismatch at index " << i;
  }
}