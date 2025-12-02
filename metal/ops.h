#ifndef METAL_OPS_H
#define METAL_OPS_H

#include <vector>

#include "gguf.h"

void init_metal();
void metal_mat_vec_mul_q4_0(std::vector<float>& o, const TensorInfo& w_tensor,
                            const GGUFFile& gguf_file,
                            const std::vector<float>& x);

#endif  // METAL_OPS_H
