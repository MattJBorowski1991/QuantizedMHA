# Quantized Multi-Head Attention (QuantizedMHA)

High-performance CUDA implementations of FlashAttention-2 with various optimizations including quantization, Tensor Cores acceleration, and warp specialization.

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

## Usage

### 1. Build
```bash
make clean && make KERNEL=<kernel_name> NVCC_ARCH=XX
```

Replace `<kernel_name>` with desired kernel (`unfused`, `fa`, `fa_warps`, `fa_tc`, or `fa_int8`; defaults to `unfused`) and `XX` with your target GPU compute capability (e.g., `80` for A100, `89` for RTX 4090, `90` for H100).

**Note:** For accurate profiling, compile one kernel variant at a time. Each compilation should have its own binary to avoid register allocation changes and ensure clean performance metrics.

### 2. (optional) Verify kernels with reference output and verify initial metrics

Runs correctness check against CPU reference and caches reference output to `.cache/` for faster subsequent runs. This step is optional but **recommended before profiling** to catch bugs early and pre-compute the reference cache.

```bash
./bin/profile_fa --warmup=5 --runs=10
```

and/or fast profile to verify if initial metrics make sense. Stop early (`Ctrl + C`).

```bash
ncu ./bin/profile_fa_int8 --warmup=0 --runs=1
```

and/or

```bash
ncu ./bin/profile_fa_tc 2>&1 | head -500 | grep "Elapsed Cycles" | tail -1 | awk '{print $NF}'
```

Available binaries: `profile_unfused`, `profile_fa`, `profile_fa_warps`, `profile_fa_tc`, `profile_fa_int8`

Options:
- `--warmup=N` - Number of warmup runs (default: 2)
- `--runs=N` - Number of profiling runs (default: 3)
- `--random` - Use random input data instead of constant values

### 3. Profile with Nsight Compute

Profile with PTX and SASS embedded:

```bash
ncu --import-source yes --set full --export profiles/ncu/fa.ncu-rep ./bin/profile_fa --kernel=fa --warmup=2 --runs=3
```

Export txt and csv summaries:
```bash
ncu --import profiles/ncu/fa_int8.ncu-rep > profiles/txt/fa_int8.txt
ncu --import profiles/ncu/fa_int8.ncu-rep --csv > profiles/csv/fa_int8.csv
```

Open `.ncu-rep` files directly in Nsight Compute.

### 4. Debug
```bash
cuda-gdb ./bin/profile_fa_warps
run
```

Or:
```bash
compute-sanitizer ./bin/profile_fa_tc
```

### 5. Notes

#### 5.1. Shared memory in `fa.cu`
Requires `(4 * max(Br*Bc, Bc*d) + 3*Br) * sizeof(float)` which equals:
- **Br=128**: ~264KB (exceeds 96KB GPU limit)
- **Br=64**: ~66KB (tight, may fail)
- **Br=32**: ~33KB (fits safely)

In case of SRAM overflow the kernel will either fail silently or generate incorrect code.

#### 5.2. Warps in FA2 and TC WMMA

- If based on fa.cu we want to implement correct TC WMMA in fa_tc.cu, we need 16×16 tiles in Q, K, V to be served by one warp.

- In the beginning in fa.cu one warp owns (warp_rows × d) = (4 × d) of Q and does a warp-stride across all columns d to calculate the matmuls, online softmax and final per-row scaled dot product.

- For this reason warp_rows needs to be a multiple of 16, let's take warp_rows = 16.

- If warp_rows = 16, then Br has to be a multiple of warp_rows:

For N = 4096, d = 2048, h = 32:
- **Br=32**: 2 warps per block, ~33KB shared mem ✓
- **Br=48**: 3 warps per block, ~50KB shared mem ✓
- **Br=64**: 4 warps per block, ~66KB shared mem (kernel fails silently) ✗


#### 5.3. Quantization Workflow

```
Inputs (float32)
  ↓ [Pre-quantize]
Q, K, V (int8)
  ↓ [QK^T: int8 @ int8]
Scores (float32 after upcast)
  ↓ [Softmax]
Probs (float32)
  ↓ [P @ V: float32 @ int8]
Output (float32)
```

**Note**: We don't quantize P because it's already a probability (small numbers 0–1).

#### 5.3. Correctness check for d=1024/16=64; Br=64; Bc=32:

(after first K, V tile pair)
1.1. Q_tile @ K_tile ^T = (64 x 64) x (64 x 32) = (64×32) with all values 64
1.2. Scale by 1/sqrt(d) = 1/8 => all values 8 == scores
and sum_exp = 32 x exp(8-8) = 32
1.3. P_tile = Softmax_rowwise(scores) = (64 x 32) with all values 1/32 
1.4. P_tile @ V_tile = (64×32) x (32x64) = (64 x 64) with all values 1 = output

(after second K, V tile pair)
sum_exp = 64
output = (64 x 64) with all values 2

..

(after 128th K,V tile pair)
sum_exp = 4096
output = (64 x 64) with all values 128

=> final output = (64 x 64 with all values) 128 / 4096





