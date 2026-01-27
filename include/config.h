#ifndef CONFIG_H
#define CONFIG_H

// Tile size for matrix operations (shared memory tiling)
constexpr int TILE = 16;

// FlashAttention block sizes (standard FA1 notation)
constexpr int Br = 32;             // Query block row size
constexpr int Bc = 32;             // Key/Value block column size

// dimensions per head
constexpr int d = 64;

// Number of CUDA streams for pipelined execution
constexpr int NSTREAMS = 2;

#endif // CONFIG_H
