# Quantized Multi-Head Attention (QuantizedMHA)

High-performance CUDA implementations of FlashAttention-2 with various optimizations including quantization, Tensor Cores acceleration, and warp specialization.

## Performance Highlights

- **Unfused baseline**: 6.44 ms total (3 kernels: Q@K^T, softmax, P@V)
- **FA_4X4**: 9.07 ms (1 fused kernel, currently occupancy-limited at 37%)
- **High occupancy unfused**: 87–100% SM utilization with 65,536 blocks
- **Identified optimizations**: 53% potential speedup via bank conflict reduction, 44% via occupancy improvements

For detailed profiling analysis, see [Profiling Results](#profiling-results) below.

## Kernels

### `unfused.cu` - Unfused Attention Components
Implements individual operations of attention separately (Q@K^T, softmax, output projection) as standalone kernels rather than fusing them into a single kernel. This modular approach allows flexibility in optimization and profiling of individual attention components.

### `fa.cu` - FlashAttention-2
A foundational FlashAttention-2 implementation with fused RoPE (Rotary Position Embeddings) that performs single-pass online softmax computation. Each block processes one Q tile with proper per-query-row normalization, using shared memory for efficient Q, K, V tile management. Each lane within warp owns a 4x4 minitile of Q in a register.

### `fa_int8.cu` - FlashAttention-2 with int8 [WIP]
Quantization applied to fa
0. Quantize float→int8 preprocessing: convert float inputs to int8 using scale/zero before main kernel
1. Dequantize on-the-fly in Q@K^T
2. Handle float x int8 for P@V matmul
3. Update SRAM layout - less mem needed for int8
4. Unchanged: softmax

### `fa_16x4.cu` - FlashAttention-2 with Tensor Cores [WIP]
Ground work for implementing Tensor Cores with double buffering in Flash Attention. 
One lane handles a 16x4 mini-tile in register so that one warp can handle a full 16 x d tile of Q.

## Project Structure

- **`mha_kernels/`** - Core attention kernel implementations
- **`extensions/`** - Language bindings (JAX, PyTorch)
- **`tests/`** - Device-side unit tests for kernels
- **`profiling/`** - Profiling tools and scripts
- **`examples/`** - Example usage scripts
- **`include/`** - Header files and configuration
- **`utils/`** - Utility functions and verification code

## Usage

### 0. Check NCU access on your device

```
bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly`
```

If you get `RmProfilingAdminOnly: 0`, it means you have access to Nvidia Performance Counters and hence to NCU.

### 1. Build
```bash
make clean && make KERNEL=<kernel_name> NVCC_ARCH=XX
```

Replace `<kernel_name>` with desired kernel (`unfused`, `fa`, `fa_int8`; defaults to `unfused`). **Note:** `fa_int8` is pending and not yet available.

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

Available binaries: `profile_unfused`, `profile_fa`, `profile_fa_int8`

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
ncu --import profiles/ncu/fa.ncu-rep > profiles/txt/fa.txt
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
Requires `(2×Br×d + (Bc+1)×d + Br×(Bc+1) + 3×Br) × sizeof(float)` (exact per-buffer allocation).

With **d=64, Bc=32**:
- **Br=128**: ~92.4 KB (exceeds 96 KB GPU limit slightly)
- **Br=64**: **50.4 KB** (current config, fits comfortably, 2 blocks/SM max)
- **Br=32**: ~29.4 KB (fits safely, allows ~3 blocks/SM if registers permit)

**Note**: Current Br=64 is optimal for future Tensor Cores (16-row warp granularity). Occupancy bottleneck is registers (64/thread), not SRAM.

#### 5.2. Warps in FA2 and TC WMMA

- If based on fa.cu we want to implement correct TC WMMA in fa_tc.cu, we need 16×16 tiles in Q, K, V to be served by one warp.

- In the beginning in fa.cu one warp owns (warp_rows × d) = (4 × d) of Q and does a warp-stride across all columns d to calculate the matmuls, online softmax and final per-row scaled dot product.

- For this reason warp_rows needs to be a multiple of 16, let's take warp_rows = 16.

- If warp_rows = 16, then Br has to be a multiple of warp_rows:

For N = 4096, d = 2048, h = 32 (past config):
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

---

## Profiling Results

Detailed profiling analysis via Nsight Compute, comparing kernel performance across unfused vs fused attention implementations.

### Run 1: Unfused vs FA_4X4 Baseline

**Summary**: Comparative profiling of three unfused attention components (`mma_A_Bt`, `softmax`, `mma_A_B`) and the fused FA_4X4 implementation.

**Key Findings**:
- **Unfused total latency**: 6.44 ms (3 kernels)
- **FA_4X4 latency**: 9.07 ms (1 kernel, 1.4× slower)
- **Occupancy**: Unfused 87–100%, FA_4X4 only 37%
- **Grid utilization**: Unfused 65,536 blocks, FA_4X4 only 64 blocks
- **Root cause**: FA_4X4 limited by register pressure (64 regs/thread) and large SRAM (102 KB)

**Top Optimization Opportunities**:
1. **Shared store bank conflicts** (est. speedup 53%) — 68.56% of stores affected
2. **Achieved occupancy** (est. speedup 44%) — 37% vs 67% theoretical
3. **Theoretical occupancy** (est. speedup 33%) — register and SRAM limits

**Full Analysis**: [profiles/md/run1/ncu_details.md](profiles/md/run1/ncu_details.md)

### Run 2: Flash Attention Kernel optimizations

**Summary**: Comparative profiling after optimizations: removed bank conflicts, introduced uniform warp/lane work, and increased sequence length N from 4096 to 8192.

**Key Findings**:
- **FA latency improvement**: Now 74% lower than unfused baseline (vs. 41% higher in Run 1).
- **Optimizations applied**: Bank conflict reduction and uniform work distribution.
- **Sequence length**: Scaled to N=8192 for larger inputs.
- **FA Memory throughput** (speed of data transfer across all device memory levels) of only ~0.4 GB/s vs 50-270 GB/s of the kernels in Unfused is a major efficiency. FA avoids redundant DRAM loads by reusing Q/K/V tiles in SRAM and L2 cache (98.84% L2 hit rate, 0.15% DRAM access).


**Status**:
- Preparing architecture for Tensor Cores (each warp handling 16×d or 32×d tiles).
- Block requirements: At least 64×d per block for Tensor Core compatibility.

**High level Analysis**: [profiles/md/run2/ncu_highlevel.md](profiles/md/run2/ncu_highlevel.md)
