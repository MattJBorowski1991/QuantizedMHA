#include <math.h>

#ifndef CONFIG_H
#define CONFIG_H

// Tile size for matrix operations in Unfused (shared memory tiling)
constexpr int TILE = 16;

// FlashAttention block sizes (standard FA1 notation)
constexpr int Br = 64;             // Query block row size
constexpr int Bc = 32;             // Key/Value block column size

// FlashAttention Warp-level tile parameters (warp-tiled matmul)
constexpr int Wr = 4;              // Number of rows handled by one Warp within a block
constexpr int Lc = 1;              // Number of columns handled by one Lane within a Warp. TODO: increase to 2


// Number of CUDA streams for pipelined execution
constexpr int NSTREAMS = 2;

// Problem size configuration
constexpr int N = 8192;             
constexpr int d_model = 1024;       // Model dimension
constexpr int h = 32;               // Number of attention heads - TODO increase back to 64 & handle sram overflow

// Validate dimensions and compute per-head dimension
static_assert(d_model % h == 0, "d_model must be divisible by h");
constexpr int d = d_model / h;      // Dimensions per head


// Validate warp-tiled matmul configuration
static_assert(d % (Lc * 32) == 0, "d must be a multiple of (Lc * 32)");
static_assert(Bc % (Lc * 32) == 0, "Bc must be a multiple of (Lc * 32)");

//Tensor Core parameters
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;


// INT8 Quantization parameters
// Scales set to preserve dynamic range: scale = max_range / 127
// For constant=1.0 inputs: int8 = round(1.0/scale), dequant = int8*scale
// Large scales preserve magnitude: 1.0 → int8≈127 → dequant≈1.0
constexpr float Q_SCALE = 1.0f;     // Q quantization scale (increase to preserve range)
constexpr float K_SCALE = 1.0f;     // K quantization scale
constexpr float V_SCALE = 1.0f;     // V quantization scale
constexpr float Q_ZERO = 0.0f;      // Q zero point
constexpr float K_ZERO = 0.0f;      // K zero point
constexpr float V_ZERO = 0.0f;      // V zero point

#endif // CONFIG_H
