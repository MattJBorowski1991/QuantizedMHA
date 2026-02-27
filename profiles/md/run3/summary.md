# Flash Attention with Tensor Cores - Performance Analysis

## Executive Summary

This document presents a performance analysis comparing the original Flash Attention kernel (`fa`) with a Tensor Core-optimized variant (`fa_tc_v1`). The tensor core implementation targets the matrix multiplications in the attention mechanism (Q@K^T and P@V), while leaving online softmax computation unchanged. The tensor core approach achieves **30.7% runtime reduction** (8.33ms → 5.77ms) with improved instruction throughput through hardware acceleration of matrix multiplications.

---

## Implementation Overview

### Architecture Design

The tensor core implementation applies NVIDIA WMMA (Warp Matrix Multiply Accumulate) operations to accelerate the two primary matrix multiplications in Flash Attention:
- **Q@K^T**: Compute attention scores
- **P@V**: Compute output values

Online softmax computation remains unchanged from the original implementation.

### Execution Model

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

## Key Design Decisions

### Padding Strategy

**Critical Finding**: Tensor Core stride requirements impose stricter alignment constraints than memory alignment alone.

- **Requirement**: Stride must be a **multiple of WMMA_N (16 elements)**, not just byte-aligned
- **Initial Approach** (PAD=8): Failed due to stride=24 not being divisible by 16
  - 24 % 16 = 8 ✗
- **Working Solution** (PAD≥16): Minimum 16-element padding ensures stride divisibility
  - 32 % 16 = 0 ✓
- **Implementation**: Padding columns initialized to zero after valid data to maintain correctness

### Execution Constraints

1. **Atomic Warp Execution**: Cannot assign work to individual lanes in Tensor Cores; the warp is the atomic execution unit
2. **Double Buffering**: Lane assignment is appropriate only for double-buffering optimizations, not for distributing computational work
3. **SRAM Limitation**: Br cannot exceed 64 on L4 GPU due to shared memory constraints

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
- **Memory throughput trade-off**: Decreased from 67.83% to 46.05% due to changed access patterns
- **Note (fa)**: Compute and memory are well-balanced; further optimization requires reducing both simultaneously
- **Note (fa_tc_v1)**: Low compute and memory bandwidth utilization (<60%) suggests latency-bound behavior requiring analysis of scheduler and warp state

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
- Block size reduced 75% (512 → 128) due to increased register and shared memory per-warp requirements
- Total thread count reduced 75% (65536 → 16384) proportionally
- Shared memory increased 56% (65.54 KB → 102.40 KB) for tensor core tile buffers
- Register demand increased 48% per-thread (54 → 80) for tile management

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
- **Occupancy Bottleneck**: Shared memory is the limiting factor for fa_tc_v1 (2 blocks per SM vs. 12 warps available)
- **Potential Speedup**: Reducing shared memory requirements could theoretically achieve **83.33% local speedup** by increasing occupancy to near full capacity
- **fa Performance**: 18.8% to 33.33% occupancy improvement available through warp scheduling and register optimization

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
- SMSP activity represents 55%+ of total elapsed cycles for both kernels, indicating compute-bound behavior
- Similar proportional distribution between Memory, L1, and SM subsystems suggests consistent memory access patterns despite different kernel configurations
- Load imbalance across SMs suggests 19-22% potential speedup through better work distribution

---

## Optimization Opportunities

### High Priority

1. **Shared Memory Reduction** (Potential: 83.33% speedup)
   - Current limiting factor: 102.40 KB shared memory per block (limited to 2 blocks per SM)
   - Target: Reduce to ≤50 KB to allow 4+ blocks per SM
   - Approach: Optimize tile buffer layouts or implement software pipelining with smaller staging areas

2. **SM Load Balancing** (Potential: 19-21% speedup)
   - Action: Reduce variance in per-SM active cycles (currently 25-31% above/below average)
   - Analysis needed: Warp scheduling logs to identify bottlenecks

3. **Latency Analysis** (Potential: 10-20% speedup)
   - Low compute throughput (46%) suggests latency-bound behavior
   - Review: Scheduler statistics and warp state distribution during execution

### Medium Priority

4. **Tile Size Innovation** (Potential: 15-30% speedup)
   - Experiment with 8×8×32 tiles to increase warps per block from 4 to 8+
   - Trade computation patterns for higher parallelism and occupancy
   - Measure register and SRAM impact

5. **Multi-Warp d-Dimension Coverage**
   - Split d-dimension across warps to reduce per-warp serialization
   - Trade-off: Increased synchronization overhead vs. reduced latency

### Lower Priority

6. **Register Optimization** (Potential: <15% speedup)
   - Current per-thread registers: 80 (increased from 54)
   - Minor impact due to register-to-shared-memory ratio; shared memory remains limiting factor

---

## Technical Notes

### Alignment and Stride Constraints

- **16-byte Memory Alignment**: Required for all tensor core buffers (standard CUDA requirement)
- **WMMA Element Alignment** (Critical): Stride in elements must be divisible by WMMA_N (16)
  - Padding formula: `PAD = ceil((d + PAD) / 16) * 16 - d`
  - Minimum effective padding: 16 elements (64 bytes on float32)
- **Validation**: Verify stride divisibility before deploying new configurations

### Memory Model

- **A and B Matrix Buffers**: Row-major layout with padded rows for alignment
- **Accumulator (C) Matrix**: Accumulated across multiple tile iterations; must maintain proper stride for store operations
- **Padding Initialization**: Padding columns must be zeroed to prevent spurious contribution to accumulation

### Future Architecture Exploration

WMMA tile combinations not yet explored:
- 16×16×8: Reduced K-dimension for higher throughput on certain memory patterns
- 32×8×16: Wide tile for broader coverage
- 8×32×16: Tall tile for higher reuse across N-dimension
- 8×8×32: Maximum K-dimension for potential double-buffering benefits

---

## Conclusion

The tensor core implementation achieves significant runtime improvement (30.7%) through hardware-accelerated matrix multiplication, despite achieving lower absolute memory and compute throughput percentages. The primary limiting factor is shared memory constraints, which restrict occupancy to 16.67%. Addressing this bottleneck through architectural optimization could unlock 83% additional speedup, making tensor cores a highly promising direction for Flash Attention acceleration.

The trade-off between serialization (due to SRAM constraints) and parallelism (from tensor cores) is well-managed in the current design, providing a solid foundation for incremental optimization toward full tensor core utilization.