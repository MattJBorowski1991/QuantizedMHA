# Flash Attention with Tensor Cores - Performance Analysis

## Executive Summary

This document presents a performance analysis comparing the original Flash Attention kernel (`fa`) with a Tensor Core-optimized variant (`fa_tc_v1`). The tensor core implementation targets the matrix multiplications in the attention mechanism (Q@K^T and P@V), while leaving online softmax computation unchanged. The tensor core approach achieves **30.7% runtime reduction** (8.33ms → 5.77ms) on `N = 8192`, `d_model = 1024` (embedding dimension) and `h = 32` (number of heads).

---

## Implementation Overview

The tensor core implementation applies WMMA to the two primary matrix multiplications in Flash Attention:
- **Q@K^T**: Compute attention scores
- **P@V**: Compute output values

Online softmax computation remains unchanged from the original implementation.

### Execution

- **Warp-Level Execution**: One warp handles WMMA_M=16 rows of Q in a serial fashion, performing one 16×16 tile multiplication at a time
- **Block Configuration**: With Br=64 (SRAM-limited) rows per block, only 4 warps execute per block, resulting in **low occupancy (16.67%)**
- **Serial vs. Parallel Work**: 
  - Non-serial WMMA operations apply only to the first warp-owned tiles: (Br × 16) of Q and (Bc × 16) of K
  - Remaining computations proceed in a serialized manner to manage SRAM constraints

### Supported Tile Sizes

The implementation uses standard 16×16×16 tiles. Future optimization opportunities include:
- **8×8×32 tile configuration**: Can increase warps per block, potentially improving occupancy from current 16.67% to higher levels
- **Multi-warp d-dimension split**: Currently one warp owns 16×d of Q serially; distributing this across multiple warps could reduce latency

---

## Observations

**Padding**: Tensor Core stride requirements impose stricter alignment constraints than memory alignment alone.

- **Requirement**: Stride must be a **multiple of WMMA_N (16 elements)**, not just byte-aligned
- **Initial Approach** (PAD=8): Failed due to stride=24 not being divisible by 16, causing outputs to be written to wrong memory addresses.

### Execution Constraints

1. **Atomic Warp Execution**: Cannot assign work to individual lanes in Tensor Cores; the warp is the atomic execution unit
2. **Double Buffering**: Lane assignment is appropriate for double-buffering optimizations, not for distributing computational work in WMMA.
3. **SRAM Limitation**: Br = 64 already puts strain on shared memory given the constraints of the L4 GPU.

---

## Performance Analysis

### Throughput Metrics

| Metric Name | Metric Unit | fa | fa_tc_v1 |
|---|---|---|---|
| DRAM Frequency | Ghz | 6.24 | 6.24 |
| SM Frequency | Mhz | 794.99 | 794.98 |
| Elapsed Cycles | cycle | 6623369 | 4585386 |
| Memory Throughput | % | 67.83 | 46.05 |
| DRAM Throughput | % | 0.15 | 0.21 |
| **Duration** | **ms** | **8.33** | **5.77** |
| L1/TEX Cache Throughput | % | 91.24 | 67.52 |
| L2 Cache Throughput | % | 2.55 | 3.59 |
| SM Active Cycles | cycle | 4917215.60 | 3123328.93 |
| Compute (SM) Throughput | % | 67.83 | 46.05 |

**Key Observations:**
- **Runtime speedup**: 30.7% reduction achieved
- **Memory & Compute throughput trade-off**: Both decreased because the kernel is now more compute-efficient (achieving faster execution with less absolute throughput), enabling further optimizations

### Launch Configuration

| Metric Name | Metric Unit | fa | fa_tc_v1 |
|---|---|---|---|
| Block Size | | 512 | 128 |
| Function Cache Configuration | | CachePreferNone | CachePreferNone |
| Grid Size | | 128 | 128 |
| Registers Per Thread | register/thread | 54 | 80 |
| Shared Memory Configuration Size | Kbyte | 65.54 | 102.40 |
| Driver Shared Memory Per Block | Kbyte/block | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block | Kbyte/block | 29.82 | 0 |
| Static Shared Memory Per Block | byte/block | 0 | 43.78 |
| # SMs | SM | 58 | 58 |
| Threads | thread | 65536 | 16384 |
| Uses Green Context | | 0 | 0 |
| Waves Per SM | | 1.10 | 1.10 |

**Configuration Differences:**
- Block size reduced 75% (512 → 128) due to the design of Flash Attention as well as Tensor Core tile sizes
- Total thread count reduced 75% (65536 → 16384) proportionally
- Shared memory arrays were changed from dynamic to static and increased to almost 44kB due to WMMA tile buffer storage required for padding
- Register demand increased 48% per-thread (54 → 80) solely due to padding (tested with PAD=0 for reference)

### Occupancy Analysis

| Metric Name | Metric Unit | fa | fa_tc_v1 |
|---|---|---|---|
| Block Limit SM | block | 24 | 24 |
| Block Limit Registers | block | 2 | 6 |
| Block Limit Shared Mem | block | 2 | 2 |
| Block Limit Warps | block | 3 | 12 |
| Theoretical Active Warps per SM | warp | 32 | 8 |
| Theoretical Occupancy | % | 66.67 | 16.67 |
| Achieved Occupancy | % | 54.13 | 15.22 |
| Achieved Active Warps Per SM | warp | 25.98 | 7.31 |

**Critical Insight:**
- **Occupancy Bottleneck**: Shared memory is still the limiting factor for the number of blocks per SM
- **Potential Speedup**: Reducing shared memory requirements could theoretically achieve **83.33% local speedup** by increasing occupancy to near full capacity

### GPU and Memory Workload Distribution

| Metric Name | Metric Unit | fa | fa % | fa_tc_v1 | fa_tc_v1 % |
|---|---|---|---|---|---|
| Average DRAM Active Cycles | cycle | 77573.33 | — | 76426.67 | — |
| Average L1 Active Cycles | cycle | 4917215.60 | — | 3123328.93 | — |
| Average L2 Active Cycles | cycle | 759107.04 | — | 903918.92 | — |
| Average SM Active Cycles | cycle | 4917215.60 | — | 3123328.93 | — |
| Average SMSP Active Cycles | cycle | 4917651.00 | — | 3123795.47 | — |
| Total DRAM Elapsed Cycles | cycle | 312161280 | 11.23% | 216112128 | 11.23% |
| Total L1 Elapsed Cycles | cycle | 383661982 | 13.81% | 265597016 | 13.80% |
| Total L2 Elapsed Cycles | cycle | 164959296 | 5.93% | 114202008 | 5.94% |
| Total SM Elapsed Cycles | cycle | 383661982 | 13.81% | 265597016 | 13.80% |
| Total SMSP Elapsed Cycles | cycle | 1534647928 | 55.22% | 1062388064 | 55.23% |
| **Total Elapsed Cycles** | **cycle** | **2779092468** | **100%** | **1923896232** | **100%** |

**Workload Balance Analysis:**
- The proportional distribution of Total Elapsed Cycles remained remarkably consistent despite the 30.7% latency improvement
- SMSP activity still represents 55%+ of total elapsed cycles for both kernels, indicating compute-bound behavior
- Similar proportional distribution between Memory, L1, and SM subsystems across both kernels suggests that tensor cores maintain consistent memory access patterns despite different execution parallelism

---

## Optimization Opportunities


1. **Tile Sizes** 
   - Experiment with 8×8×32 tiles to increase warps per block from 4 to 8+
   - Trade computation patterns for higher parallelism and occupancy
   - Measure register and SRAM impact

2. **Multi-Warp d-Dimension Coverage**
   - Split d-dimension across warps to reduce per-warp serialization
   - Trade-off: Increased synchronization overhead vs. reduced latency

3. **Latency Analysis** (Potential: 10-20% speedup)
   - Low compute throughput (46%) suggests latency-bound behavior
   - Review: Scheduler statistics and warp state distribution during execution

4. **Register Optimization** (Potential: <15% speedup)
   - Current per-thread registers: 80 (increased from 54)
   - Minor impact due to shared memory remaining the limiting factor

---

## Conclusion

The tensor core implementation achieves significant runtime improvement (30.7%), despite achieving lower absolute memory and compute throughput percentages. The primary limiting factor now is the low occupancy (16.67%), due to algorithm design - one warp handles 16 rows, one block handles 64 rows (of Q). Addressing this bottleneck through architectural optimization could unlock significant speedup.

The trade-off between serialization (due to SRAM constraints) and parallelism (from tensor cores) is well-managed in the current design, providing a solid foundation for incremental optimization toward full tensor core utilization.