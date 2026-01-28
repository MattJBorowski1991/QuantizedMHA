#ifndef CONFIG_H
#define CONFIG_H

// Tile size for matrix operations (shared memory tiling)
constexpr int TILE = 16;

// FlashAttention block sizes (standard FA1 notation)
constexpr int Br = 32;             // Query block row size
constexpr int Bc = 32;             // Key/Value block column size

// Number of CUDA streams for pipelined execution
constexpr int NSTREAMS = 2;

// Problem size configuration
constexpr int N = 4096;             // Sequence length
constexpr int d_model = 2048;       // Model dimension
constexpr int h = 32;               // Number of attention heads

// Validate dimensions and compute per-head dimension
static_assert(d_model % h == 0, "d_model must be divisible by h");
constexpr int d = d_model / h;      // Dimensions per head

// INT8 Quantization parameters
constexpr float Q_SCALE = 0.01f;    // Q quantization scale
constexpr float K_SCALE = 0.01f;    // K quantization scale
constexpr float V_SCALE = 0.01f;    // V quantization scale
constexpr float Q_ZERO = 0.0f;      // Q zero point
constexpr float K_ZERO = 0.0f;      // K zero point
constexpr float V_ZERO = 0.0f;      // V zero point

#endif // CONFIG_H
