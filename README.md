# Quantized Multi-Head Attention (QuantizedMHA)

High-performance CUDA implementations of FlashAttention-2 with various optimizations including quantization, Tensor Cores acceleration, and warp specialization.

## Usage

### 1. Build
```bash
make clean && make NVCC_ARCH=86
```

Replace `86` with your target GPU compute capability (e.g., `80` for A100, `89` for RTX 4090, `90` for H100).

### 2. Run Kernels
```bash
./bin/main --kernel=fa --warmup=5 --runs=10
```

Available kernels: `unfused`, `fa`, `fa_warps`, `fa_tc`, `fa_int8`

Options:
- `--kernel=<NAME>` - Select kernel to run
- `--warmup=N` - Number of warmup runs (default: 5)
- `--runs=N` - Number of profiling runs (default: 10)
- `--random` - Use random input data instead of constant values

### 3. Profile with Nsight Compute

Profile with PTX and SASS embedded:

```bash
ncu --import-source yes --set full --export profiles/ncu/fa_int8.ncu-rep ./bin/main --kernel=fa_int8 --warmup=5 --runs=10
```

Export txt and csv summaries:
```bash
ncu --import profiles/ncu/fa_int8.ncu-rep > profiles/txt/fa_int8.txt
ncu --import profiles/ncu/fa_int8.ncu-rep --csv > profiles/csv/fa_int8.csv
```

Open `.ncu-rep` files directly in Nsight Compute.

### 4. Debug
```bash
cuda-gdb ./bin/main
run --kernel=fa_warps
```

Or with cuda-memcheck:
```bash
cuda-memcheck ./bin/main --kernel=fa
```


## Kernels

### `unfused.cu` - Unfused Attention Components
Implements individual operations of attention separately (Q@K^T, softmax, output projection) as standalone kernels rather than fusing them into a single kernel. This modular approach allows flexibility in optimization and profiling of individual attention components.

### `fa.cu` - Standard FlashAttention-2
A foundational FlashAttention-2 implementation with fused RoPE (Rotary Position Embeddings) that performs single-pass online softmax computation. Each block processes one Q tile with proper per-query-row normalization, using shared memory for efficient Q, K, V tile management.

### `fa_warps.cu` - FlashAttention-2 with Warp Specialization
A variant of FlashAttention-2 that assigns each warp in a block to handle one query row independently. This parallelization strategy uses shared memory efficiently to store multiple Q rows and shared K, V tiles, reducing synchronization overhead.

### `fa_tc.cu` - FlashAttention-2 with Tensor Cores
Optimized FlashAttention-2 kernel that leverages NVIDIA Tensor Cores (WMMA) and warp-level specialization for faster matrix operations. Uses half-precision (FP16) computations with double buffering (cp.async) to hide memory latency while maintaining numerical stability through float accumulation.

### `fa_int8.cu` - FlashAttention-2 with Tensor Cores & INT8 Quantization
Combines Tensor Core acceleration with INT8 quantization to reduce memory footprint and increase throughput. Supports quantization/dequantization with per-tensor scales and zero points for Q, K, V, enabling efficient inference on quantized attention computations.


## Project Structure

- **`mha_kernels/`** - Core attention kernel implementations
- **`extensions/`** - Language bindings (JAX, PyTorch)
- **`tests/`** - Device-side unit tests for kernels
- **`profiling/`** - Profiling tools and scripts
- **`examples/`** - Example usage scripts
- **`include/`** - Header files and configuration
- **`utils/`** - Utility functions and verification code
