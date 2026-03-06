# Quantized Multi-Head Attention (QuantizedMHA)

High-performance CUDA implementations of FlashAttention-2 with various optimizations including quantization, Tensor Cores and warp specialization.

## Performance Highlights

**Flash Attention** = **FA**, **Tensor Cores** = **TC**.

| FA | TC | Kernel | Time (ms) | Setup | Notes |
|----|----|--------|-----------|-------|-------|
| No | No | [unfused](mha_kernels/unfused.cu) | 14.4 | 3 kernels: Q@K^T (6.5), softmax (2.2), P@V (5.7) | Baseline |
| Yes | No | [fa](mha_kernels/fa.cu) | 8.33 | 1 fused kernel | - |
| Yes | Yes | [fa_tc_v1a](mha_kernels/fa_tc_v1a.cu) | 5.77 | 1 warp owns 16×d of Q | v. low occupancy |
| Yes | Yes | [fa_tc_v1b](mha_kernels/fa_tc_v1b.cu) | 6.00 | 1 warp owns 8×d of Q | - |
| Yes | Yes | [fa_tc_v2](mha_kernels/fa_tc_v2.cu) | 8.29 | 2 warps own 8×d of Q | Bank conflicts |
| Yes | Yes | [fa_tc_v2a](mha_kernels/fa_tc_v2a.cu) | 6.25 | 2 warps own 8×d of Q | Padding added |
| Yes | Yes | [fa_tc_v2b](mha_kernels/fa_tc_v2b.cu) | 9.60 | 2 warps own 8×d of Q | Swizzling added |

For detailed profiling analysis, see [Profiling Results](#profiling-results) below.

## Profiling Results

Detailed profiling analysis via Nsight Compute, comparing kernel performance across unfused vs fused attention implementations.

### Run 1: Unfused vs FA

Comparative profiling of three unfused attention components (`mma_A_Bt`, `softmax`, `mma_A_B`) and the fused FA implementation.

**Detailed Analysis**: [profiles/md/run1/ncu_details.md](profiles/md/run1/ncu_details.md)

### Run 2: Unfused vs FA after optimizations

Removed bank conflicts, unified warp/lane work, reduced Shared Memory usage and register pressure. 

**Detailed Analysis**: [profiles/md/run2/ncu_details.md](profiles/md/run2/ncu_details.md)

### Run 3a: Tensor Cores: first implementation

First implementation with the standard tile size of `16x16x16` and one warp owning `16 x d` of `Q` in a serialized way.

Kernels profiled: [fa.cu](../../../mha_kernels/fa.cu) and [fa_tc_v1.cu](../../../mha_kernels/fa_tc_v1.cu).
.
**High Level Comparison**: [profiles/md/run3a/ncu_details.md](profiles/md/run3a/ncu_high_level.md)

### Run 3b: Tensor Cores: Nsight Compute analysis

Kernels profiled: [fa_tc_v1.cu](../../../mha_kernels/fa_tc_v1.cu).

**Detailed Analysis**: [profiles/md/run3b/ncu_details.md](profiles/md/run3b/ncu_details.md)

### Run 4: Tensor Cores: more warps working

Kernels profiled: [fa_tc_v1a.cu](../../../mha_kernels/fa_tc_v1a.cu) and [fa_tc_v2.cu](../../../mha_kernels/fa_tc_v2.cu).

Changed WMMA tile size to 8x32x16. Distributed warp work across d-dimension of Q to two warps.

**Detailed Analysis**: [profiles/md/run4/ncu_details.md](profiles/md/run4/ncu_details.md)

### Run 5: Tensor Cores: remove bank conflicts

Optimized SRAM usage in to enable `PAD=8` or `PAD=16` and eliminate bank conflicts.

Kernels profiled: [fa_tc_v2.cu](../../../mha_kernels/fa_tc_v2.cu) and [fa_tc_v2a.cu](../../../mha_kernels/fa_tc_v2a.cu).

**Detailed Analysis**: [profiles/md/run5/ncu_details.md](profiles/md/run5/ncu_details.md)


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

compute-sanitizer --tool memcheck ./bin/profile_fa_tc_v1 --warmup=1 --runs=1 2>&1 | head -150

```
