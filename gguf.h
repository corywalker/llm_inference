#ifndef GGUF_H
#define GGUF_H

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// GGUF Magic number and version constants
const uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" in ASCII
const uint32_t GGUF_VERSION = 3;         // Current version we support

enum class GGUFType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
};

enum class GGUFTensorType : uint32_t {
  F32 = 0,
  F16 = 1,
  Q4_0 = 2,
  Q4_1 = 3,
  Q5_0 = 6,
  Q5_1 = 7,
  Q8_0 = 8,
  Q8_1 = 9,
  Q2_K = 10,
  Q3_K = 11,
  Q4_K = 12,
  Q5_K = 13,
  Q6_K = 14,
  Q8_K = 15,
};

std::string tensorTypeToString(uint32_t type);

struct GGUFHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_kv_count;
};

class GGUFValue {
 public:
  GGUFType type;
  union {
    uint8_t u8;
    int8_t i8;
    uint16_t u16;
    int16_t i16;
    uint32_t u32;
    int32_t i32;
    float f32;
    uint64_t u64;
    int64_t i64;
    double f64;
    bool b;
  } scalar;
  std::string str;
  std::vector<GGUFValue> arr;

  GGUFValue() : type(GGUFType::UINT32) { scalar.u32 = 0; }
};

using GGUFValueMap = std::map<std::string, GGUFValue>;

struct TensorInfo {
  std::string name;
  std::vector<uint64_t> shape;
  uint64_t total_elements;
  uint32_t tensor_type;
  uint64_t tensor_offset;
};

class GGUFFile {
 public:
  GGUFFile(const std::string& filename);
  GGUFFile(const uint8_t* data, size_t size);
  ~GGUFFile();

  const GGUFHeader& get_header() const { return header; }
  const GGUFValueMap& get_metadata() const { return metadata; }
  const std::vector<TensorInfo>& get_tensor_infos() const {
    return tensor_infos;
  }

  void print_metadata() const;
  void print_tensor_infos() const;
  void read_tensor_data(const TensorInfo& tensor, void* data,
                        size_t size) const;
  void read_tensor_data_region(const TensorInfo& tensor, size_t data_offset,
                               void* data, size_t size) const;
  const uint8_t* get_tensor_data(const TensorInfo& tensor) const;

 private:
  class GGUFReader;
  GGUFReader* reader_;
  const uint8_t* file_data_;
  size_t file_size_;

  GGUFHeader header;
  GGUFValueMap metadata;
  std::vector<TensorInfo> tensor_infos;
  size_t data_section_start;

  void load();
};

float f16_to_f32(uint16_t f16);
float dequantize_q4_0(uint8_t q4, float d);
void print_value(const GGUFValue& value);

#endif  // GGUF_H
