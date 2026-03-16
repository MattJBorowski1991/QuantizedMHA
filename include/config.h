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

#endif // CONFIG_H
