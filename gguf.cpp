#include "gguf.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "common.h"

// The following functions are copied from llama.cpp/ggml/src/ggml-impl.h
// START functions from llama.cpp.

// MIT License
//
// Copyright (c) 2023-2024 The ggml authors

static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&            \
    (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
      sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                          : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

// END functions from llama.cpp.

// Helper function to convert F16 to F32
// Reference impl is at llama.cpp/ggml/src/ggml-impl.h
// ggml_compute_fp16_to_fp32()
float f16_to_f32(uint16_t f16) { return ggml_compute_fp16_to_fp32(f16); }

class GGUFFile::GGUFReader {
 public:
  GGUFReader(const char* filename) : is_mmap(true) {
    fd = open(filename, O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open GGUF file. Check path.");
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      close(fd);
      throw std::runtime_error("Failed to get file size");
    }
    fileSize = sb.st_size;

    data = static_cast<uint8_t*>(
        mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE, fd, 0));
    if (data == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to mmap file");
    }
    currentPos = 0;
  }

  GGUFReader(const uint8_t* buffer, size_t size)
      : fd(-1),
        data(const_cast<uint8_t*>(buffer)),
        fileSize(size),
        currentPos(0),
        is_mmap(false) {}

  ~GGUFReader() {
    if (is_mmap && data != MAP_FAILED) {
      munmap(data, fileSize);
    }
    if (fd != -1) {
      close(fd);
    }
  }

  const uint8_t* getDataPtr() const { return data; }
  size_t getFileSize() const { return fileSize; }

  template <typename T>
  T read() {
    T value;
    if (currentPos + sizeof(T) > fileSize) {
      throw std::runtime_error("Read beyond end of file");
    }
    memcpy(&value, data + currentPos, sizeof(T));
    currentPos += sizeof(T);
    return value;
  }

  std::string readString() {
    uint64_t length = read<uint64_t>();
    if (length > fileSize) {
      throw std::runtime_error("Invalid string length: " +
                               std::to_string(length));
    }
    if (currentPos + length > fileSize) {
      throw std::runtime_error("String length exceeds file size");
    }
    std::string result(reinterpret_cast<char*>(data + currentPos), length);
    currentPos += length;
    return result;
  }

  GGUFValue readValue(GGUFType type, bool skipArrays = false);

  size_t getPosition() const { return currentPos; }

 private:
  int fd;
  uint8_t* data;
  size_t fileSize;
  size_t currentPos;
  bool is_mmap;
};

GGUFValue GGUFFile::GGUFReader::readValue(GGUFType type, bool skipArrays) {
  GGUFValue value;
  value.type = type;

  switch (type) {
    case GGUFType::UINT8:
      value.scalar.u8 = read<uint8_t>();
      break;
    case GGUFType::INT8:
      value.scalar.i8 = read<int8_t>();
      break;
    case GGUFType::UINT16:
      value.scalar.u16 = read<uint16_t>();
      break;
    case GGUFType::INT16:
      value.scalar.i16 = read<int16_t>();
      break;
    case GGUFType::UINT32:
      value.scalar.u32 = read<uint32_t>();
      break;
    case GGUFType::INT32:
      value.scalar.i32 = read<int32_t>();
      break;
    case GGUFType::FLOAT32:
      value.scalar.f32 = read<float>();
      break;
    case GGUFType::UINT64:
      value.scalar.u64 = read<uint64_t>();
      break;
    case GGUFType::INT64:
      value.scalar.i64 = read<int64_t>();
      break;
    case GGUFType::FLOAT64:
      value.scalar.f64 = read<double>();
      break;
    case GGUFType::BOOL:
      value.scalar.b = read<bool>();
      break;
    case GGUFType::STRING:
      value.str = readString();
      break;
    case GGUFType::ARRAY: {
      uint32_t type_int = read<uint32_t>();
      GGUFType element_type = static_cast<GGUFType>(type_int);
      uint64_t count = read<uint64_t>();
      value.scalar.u64 = count;
      value.str = std::to_string(type_int);
      if (!skipArrays) {
        value.arr.reserve(count);
        for (uint64_t i = 0; i < count; i++) {
          value.arr.push_back(readValue(element_type, skipArrays));
        }
      } else {
        // Efficiently skip array data
      }
      break;
    }
    default:
      throw std::runtime_error("Unsupported GGUF value type");
  }
  return value;
}

GGUFFile::GGUFFile(const std::string& filename)
    : reader_(new GGUFReader(filename.c_str())) {
  load();
  file_data_ = reader_->getDataPtr();
  file_size_ = reader_->getFileSize();
}

GGUFFile::GGUFFile(const uint8_t* data, size_t size)
    : reader_(new GGUFReader(data, size)) {
  load();
  file_data_ = reader_->getDataPtr();
  file_size_ = reader_->getFileSize();
}

GGUFFile::~GGUFFile() { delete reader_; }

void GGUFFile::load() {
  header = reader_->read<GGUFHeader>();
  if (header.magic != GGUF_MAGIC) {
    throw std::runtime_error("Invalid GGUF magic number");
  }

  for (uint64_t i = 0; i < header.metadata_kv_count; ++i) {
    std::string key = reader_->readString();
    GGUFType type = static_cast<GGUFType>(reader_->read<uint32_t>());
    metadata[key] = reader_->readValue(type);
  }

  for (uint64_t i = 0; i < header.tensor_count; ++i) {
    TensorInfo info;
    info.name = reader_->readString();
    uint32_t dimensions = reader_->read<uint32_t>();
    info.shape.resize(dimensions);
    info.total_elements = 1;
    for (uint32_t j = 0; j < dimensions; ++j) {
      info.shape[j] = reader_->read<uint64_t>();
      info.total_elements *= info.shape[j];
    }
    info.tensor_type = reader_->read<uint32_t>();
    info.tensor_offset = reader_->read<uint64_t>();
    tensor_infos.push_back(info);
  }

  size_t current_pos = reader_->getPosition();
  size_t alignment = 32;
  data_section_start = (current_pos + alignment - 1) & ~(alignment - 1);
}

void GGUFFile::print_metadata() const {
  LOG_VERBOSE("\nMetadata:\n");
  for (const auto& kv : metadata) {
    std::cout << kv.first << " = ";
    print_value(kv.second);
    std::cout << std::endl;
  }
}

void GGUFFile::print_tensor_infos() const {
  LOG_VERBOSE("\nTensors:\n");
  for (const auto& info : tensor_infos) {
    std::cout << info.name << ": shape = [";
    for (size_t j = 0; j < info.shape.size(); j++) {
      if (j > 0) std::cout << ", ";
      std::cout << info.shape[j];
    }
    std::cout << ", elements = " << info.total_elements;
    std::cout << ", type = " << tensorTypeToString(info.tensor_type);
    std::cout << ", offset = " << info.tensor_offset << "\n";
  }
}

float dequantize_q4_0(uint8_t q4, float d) {
  // The 4-bit value is in [0, 15], we subtract 8 to get signed value in [-8, 7]
  return (static_cast<int>(q4) - 8) * d;
}

void GGUFFile::read_tensor_data(const TensorInfo& tensor, void* data,
                                size_t size) const {
  size_t tensor_abs_pos = data_section_start + tensor.tensor_offset;
  if (tensor_abs_pos + size > file_size_) {
    throw std::runtime_error("Read beyond end of file");
  }
  memcpy(data, file_data_ + tensor_abs_pos, size);
}

void GGUFFile::read_tensor_data_region(const TensorInfo& tensor,
                                       size_t data_offset, void* data,
                                       size_t size) const {
  size_t tensor_abs_pos =
      data_section_start + tensor.tensor_offset + data_offset;
  if (tensor_abs_pos + size > file_size_) {
    throw std::runtime_error("Read beyond end of file");
  }
  memcpy(data, file_data_ + tensor_abs_pos, size);
}

const uint8_t* GGUFFile::get_tensor_data(const TensorInfo& tensor) const {
  return file_data_ + data_section_start + tensor.tensor_offset;
}

std::string tensorTypeToString(uint32_t type) {
  switch (static_cast<GGUFTensorType>(type)) {
    case GGUFTensorType::F32:
      return "F32";
    case GGUFTensorType::F16:
      return "F16";
    case GGUFTensorType::Q4_0:
      return "Q4_0";
    // ... other types ...
    default:
      return "UNKNOWN";
  }
}

void print_value(const GGUFValue& value) {
  switch (value.type) {
    case GGUFType::UINT8:
      std::cout << static_cast<uint32_t>(value.scalar.u8);
      break;
    case GGUFType::INT8:
      std::cout << static_cast<int32_t>(value.scalar.i8);
      break;
    case GGUFType::UINT16:
      std::cout << value.scalar.u16;
      break;
    case GGUFType::INT16:
      std::cout << value.scalar.i16;
      break;
    case GGUFType::UINT32:
      std::cout << value.scalar.u32;
      break;
    case GGUFType::INT32:
      std::cout << value.scalar.i32;
      break;
    case GGUFType::FLOAT32:
      std::cout << value.scalar.f32;
      break;
    case GGUFType::UINT64:
      std::cout << value.scalar.u64;
      break;
    case GGUFType::INT64:
      std::cout << value.scalar.i64;
      break;
    case GGUFType::FLOAT64:
      std::cout << value.scalar.f64;
      break;
    case GGUFType::BOOL:
      std::cout << (value.scalar.b ? "true" : "false");
      break;
    case GGUFType::STRING:
      std::cout << '"' << value.str << '"';
      break;
    case GGUFType::ARRAY:
      std::cout << "[";
      for (size_t i = 0; i < std::min((size_t)5, value.arr.size()); i++) {
        if (i > 0) std::cout << ", ";
        print_value(value.arr[i]);
      }
      if (value.arr.size() > 5) {
        std::cout << ", ...";
      }
      std::cout << "]";
      break;
    default:
      std::cout << "Unknown type";
  }
}
