#include <metal_stdlib>
using namespace metal;

// Q4_0 block size
constant int BLOCK_SIZE = 32;

struct TensorInfo {
    int rows;
    int cols;
};

// Dequantize Q4_0 block
// Each block has 2 bytes (float16 scale) + 16 bytes (32 x 4-bit values)
// Total 18 bytes per block.
//
// Note: This kernel assumes the input `weights` points to the start of the row.
//
// weights: pointer to the raw GGUF data for the tensor
// input: the float32 input vector (x)
// output: the float32 output vector (o)
//
// We launch one thread per row.
kernel void mat_vec_mul_q4_0(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant TensorInfo& info [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= info.rows) {
        return;
    }

    const int n_cols = info.cols;
    const int blocks_per_row = (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int bytes_per_block = 18; // 2 (scale) + 16 (quants)

    // Offset to the start of this row's data in weights
    long row_offset = (long)id * blocks_per_row * bytes_per_block;
    device const uchar* row_ptr = weights + row_offset;

    float sum = 0.0f;

    for (int i = 0; i < blocks_per_row; ++i) {
        // Each block:
        // 0-1: float16 scale
        // 2-17: 32 x 4-bit values

        device const uchar* block_ptr = row_ptr + i * bytes_per_block;
        
        // Read scale (float16)
        half scale_h = *(device const half*)(block_ptr);
        float scale = (float)scale_h;

        device const uchar* quants = block_ptr + 2;

        for (int j = 0; j < BLOCK_SIZE/2; ++j) {
            uchar q_pair = quants[j];
            
            // Low nibble
            uchar q0 = q_pair & 0x0F;
            float v0 = (float)(q0 - 8) * scale;
            
            // High nibble
            uchar q1 = (q_pair >> 4) & 0x0F;
            float v1 = (float)(q1 - 8) * scale;

            int col_idx = i * BLOCK_SIZE + j;
            if (col_idx < n_cols) {
                sum += v0 * input[col_idx];
            }
            
            if (col_idx + BLOCK_SIZE/2 < n_cols) {
                sum += v1 * input[col_idx + BLOCK_SIZE/2];
            }
        }
    }

    output[id] = sum;
}
