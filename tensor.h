#ifndef TENSOR_H
#define TENSOR_H

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

typedef std::vector<float> tensor_1;
typedef std::vector<tensor_1> tensor_2;
typedef std::vector<tensor_2> tensor_3;

// Generic tensor printing function
inline void print_tensor_generic(const tensor_1& data,
                                 std::vector<int64_t> shape,
                                 const std::string& name,
                                 bool full_dump = false) {
  std::cout << name;

  while (shape.size() < 4) {
    shape.push_back(1);
  }

  const int64_t ne[4] = {shape[0], shape[1], shape[2], shape[3]};

  std::cout << " = {" << ne[0];
  for (size_t i = 1; i < 4; ++i) {
    std::cout << ", " << ne[i];
  }
  std::cout << "}" << std::endl;

  auto get_float_value = [&](size_t i0, size_t i1, size_t i2, size_t i3) {
    size_t i =
        i3 * ne[2] * ne[1] * ne[0] + i2 * ne[1] * ne[0] + i1 * ne[0] + i0;
    return data[i];
  };

  float sum = std::accumulate(data.begin(), data.end(), 0.0f);

  const int64_t n = 3;

  for (int64_t i3 = 0; i3 < ne[3]; i3++) {
    std::cout << "    [" << std::endl;
    for (int64_t i2 = 0; i2 < ne[2]; i2++) {
      if (!full_dump && i2 == n && ne[2] > 2 * n) {
        std::cout << "     ...," << std::endl;
        i2 = ne[2] - n;
      }
      std::cout << "     [" << std::endl;
      for (int64_t i1 = 0; i1 < ne[1]; i1++) {
        if (!full_dump && i1 == n && ne[1] > 2 * n) {
          std::cout << "      ...," << std::endl;
          i1 = ne[1] - n;
        }
        std::cout << "      [";
        for (int64_t i0 = 0; i0 < ne[0]; i0++) {
          if (!full_dump && i0 == n && ne[0] > 2 * n) {
            std::cout << "..., ";
            i0 = ne[0] - n;
          }
          const float v = get_float_value(i0, i1, i2, i3);
          std::cout << std::fixed << std::setprecision(4) << std::setw(12) << v;
          if (i0 < ne[0] - 1) std::cout << ", ";
        }
        std::cout << "]," << std::endl;
      }
      std::cout << "     ]," << std::endl;
    }
    std::cout << "    ]" << std::endl;
  }
  std::cout << "    sum = " << std::fixed << std::setprecision(6) << sum
            << std::endl;
  if (std::isnan(sum)) {
    std::cerr << "encountered NaN - aborting" << std::endl;
    exit(0);
  }
}

// Function to print a 1D tensor (vector of floats)
inline void print_tensor(const tensor_1& t, const std::string& name,
                         bool full_dump = false) {
  if (t.empty()) {
    std::cout << name << ": []" << std::endl;
    return;
  }
  print_tensor_generic(t, {(int64_t)t.size()}, name, full_dump);
}

// Overload to print a 1D tensor with a specific shape
inline void print_tensor(const tensor_1& t, const std::vector<uint64_t>& shape,
                         const std::string& name, bool full_dump = false) {
  if (t.empty()) {
    std::cout << name << ": []" << std::endl;
    return;
  }
  std::vector<int64_t> shape64;
  for (uint64_t dim : shape) {
    shape64.push_back(static_cast<int64_t>(dim));
  }
  print_tensor_generic(t, shape64, name, full_dump);
}

// Function to print a 2D tensor (vector of vectors of floats)
inline void print_tensor(const tensor_2& t, const std::string& name,
                         bool full_dump = false) {
  if (t.empty() || t[0].empty()) {
    std::cout << name << ": [[]]" << std::endl;
    return;
  }
  tensor_1 flat;
  for (const auto& row : t) {
    flat.insert(flat.end(), row.begin(), row.end());
  }
  print_tensor_generic(flat, {(int64_t)t[0].size(), (int64_t)t.size()}, name,
                       full_dump);
}

// Function to print a 3D tensor (vector of vectors of vectors of floats)
inline void print_tensor(const tensor_3& t, const std::string& name,
                         bool full_dump = false) {
  if (t.empty() || t[0].empty() || t[0][0].empty()) {
    std::cout << name << ": [[[]]]" << std::endl;
    return;
  }
  tensor_1 flat;
  for (const auto& slice : t) {
    for (const auto& row : slice) {
      flat.insert(flat.end(), row.begin(), row.end());
    }
  }
  print_tensor_generic(
      flat, {(int64_t)t[0][0].size(), (int64_t)t[0].size(), (int64_t)t.size()},
      name, full_dump);
}

inline tensor_3 reshape_3d(const tensor_2& input, uint32_t dim1, uint32_t dim2,
                           uint32_t dim3) {
  tensor_3 output(dim1, tensor_2(dim2, tensor_1(dim3)));
  for (uint32_t i = 0; i < dim1; ++i) {
    for (uint32_t j = 0; j < dim2; ++j) {
      for (uint32_t k = 0; k < dim3; ++k) {
        output[i][j][k] = input[i][j * dim3 + k];
      }
    }
  }
  return output;
}

#endif  // TENSOR_H
