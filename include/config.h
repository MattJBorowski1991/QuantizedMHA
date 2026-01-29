#ifndef CONFIG_H
#define CONFIG_H

// Tile size for matrix operations (shared memory tiling)
constexpr int TILE = 16;

// FlashAttention block sizes (standard FA1 notation)
constexpr int Br = 64;             // Query block row size
constexpr int Bc = 32;             // Key/Value block column size

// Number of CUDA streams for pipelined execution
constexpr int NSTREAMS = 2;

// Problem size configuration
constexpr int N = 4096;             // Sequence length
constexpr int d_model = 1024;       // Model dimension
constexpr int h = 16;               // Number of attention heads

// Validate dimensions and compute per-head dimension
static_assert(d_model % h == 0, "d_model must be divisible by h");
constexpr int d = d_model / h;      // Dimensions per head

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
