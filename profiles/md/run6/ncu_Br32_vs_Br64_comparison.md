# Br=32 vs Br=64 Profiling Comparison

High-level NCU comparison of two fa_tc_v3 configurations: **Br=64 with 8×2=16 warps/block** vs **Br=32 with 4×2=8 warps/block**, with warp work distributed across both the block row dimension (Br) and embedding dimension (d). Both kernels compiled with `-maxrregcount=70` to control register pressure.

> **Compilation Flag**: Both kernels compiled with `-maxrregcount=70` to limit register pressure and allow more blocks resident on GPU.

## GPU Speed Of Light Throughput

| Metric Name | Br=32 | Br=64 | Diff (%) |
|---|---|---|---|
| **DRAM Frequency (Ghz)** | 6.24 | 6.24 | 0.00 |
| **SM Frequency (Mhz)** | 824.93 | 798.98 | -3.14 |
| **Elapsed Cycles** | 8,631,167 | 9,783,995 | 13.37 |
| **Memory Throughput (%)** | **56.27** | 47.32 | **-8.95** |
| **DRAM Throughput (%)** | 0.22 | 0.15 | -0.07 |
| **Duration (ms)** | **10.33** | 12.10 | **-17.2%** 🎯 |
| **L1/TEX Cache Throughput (%)** | 66.34 | 66.06 | -0.28 |
| **L2 Cache Throughput (%)** | **7.80** | 2.91 | **-62.7%** ⚠️ |
| **SM Active Cycles** | 7,262,302.97 | 6,917,795.10 | -4.75 |
| **Compute (SM) Throughput (%)** | 55.15 | 47.32 | **-7.83** ⚠️ |

### Nsight Compute Observations

**Br=32:**
> OPT - This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

**Br=64:**
> OPT - This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

---

## Launch Statistics

| Metric Name | Br=32 | Br=64 |
|---|---|---|
| **Block Size** | 256 | 512 |
| Function Cache Configuration | CachePreferNone | CachePreferNone |
| **Grid Size** | 256 | 128 |
| **Registers Per Thread** | 60 | 63 |
| Shared Memory Config Size (KB) | 102.40 | 102.40 |
| Driver Shared Memory Per Block (KB) | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block (B) | 0 | 0 |
| **Static Shared Memory Per Block (KB)** | 19.01 | **37.12** ⚠️ |
| # SMs | 58 | 58 |
| Threads | 65536 | 65536 |
| Uses Green Context | 0 | 0 |
| Waves Per SM | 1.10 | 1.10 |

### Key Observations

- **Br=64** uses nearly **2x static shared memory** (37.12 KB vs 19.01 KB)
- **Br=32** has **2x grid size** (256 vs 128) → better scheduling flexibility  
- **Br=64** has higher registers per thread (63 vs 60)

---

## Occupancy

| Metric Name | Br=32 | Br=64 |
|---|---|---|
| Block Limit SM | 24 | 24 |
| **Block Limit Registers** | 4 | **2** ⚠️ |
| **Block Limit Shared Mem** | 5 | **2** ⚠️ |
| **Block Limit Warps** | 6 | **3** ⚠️ |
| Theoretical Active Warps per SM | 32 | 32 |
| Theoretical Occupancy (%) | 66.67 | 66.67 |
| **Achieved Occupancy (%)** | 53.78 | **57.83** ⚠️ |
| **Achieved Active Warps Per SM** | 25.82 | **27.76** ⚠️ |

### Nsight Compute Observations

**Br=32:**
> OPT - **Est. Local Speedup: 19.32%**
> The difference between calculated theoretical (66.7%) and measured achieved occupancy (53.8%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution.

> OPT - **Est. Local Speedup: 33.33%**  
> The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of required registers.

**Br=64:**
> OPT - **Est. Local Speedup: 33.33%**
> The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of required registers AND the required amount of shared memory.

---

## GPU and Memory Workload Distribution

| Metric Name | Br=32 | Br=64 | Diff (%) |
|---|---|---|---|
| **Average DRAM Active Cycles** | 139,861.33 | 113,037.33 | -19.16 |
| Total DRAM Elapsed Cycles | 387,004,416 | 453,372,928 | +17.16 |
| Average L1 Active Cycles | 7,262,302.97 | 6,917,795.10 | -4.75 |
| Total L1 Elapsed Cycles | 496,552,498 | 560,144,070 | +12.81 |
| **Average L2 Active Cycles** | **6,040,280.58** | 4,086,572.12 | **-32.28** ⚠️ |
| Total L2 Elapsed Cycles | 206,890,728 | 239,582,520 | +15.80 |
| Average SM Active Cycles | 7,262,302.97 | 6,917,795.10 | -4.75 |
| Total SM Elapsed Cycles | 496,552,498 | 560,144,070 | +12.81 |
| Average SMSP Active Cycles | 7,182,420.75 | 6,813,111.07 | -5.14 |
| Total SMSP Elapsed Cycles | 1,986,209,992 | 2,240,576,280 | +12.80 |

### Nsight Compute Observations

**Br=32:**
> OPT - **Est. Speedup: 12.97%** (SM load imbalance)
> One or more SMs have a much higher number of active cycles than the average. Maximum is 15.29% above average.

> OPT - **Est. Speedup: 13.62%** (SMSP load imbalance)  
> One or more SMSPs have a much higher number of active cycles than the average. Maximum is 16.23% above average.

> OPT - **Est. Speedup: 12.97%** (L1 load imbalance)
> One or more L1 Slices have a much higher number of active cycles than the average. Maximum is 15.29% above average.

> OPT - **Est. Speedup: 5.225%** (L2 load imbalance)
> One or more L2 Slices have a much higher number of active cycles than the average. Maximum is 7.46% above average.

**Br=64:**
> OPT - **Est. Speedup: 20.85%** (SM load imbalance ⚠️)
> One or more SMs have a much higher number of active cycles than the average. Maximum is **29.11%** above average.

> OPT - **Est. Speedup: 21.45%** (SMSP load imbalance ⚠️)
> One or more SMSPs have a much higher number of active cycles than the average. Maximum is **30.40%** above average.

> OPT - **Est. Speedup: 20.85%** (L1 load imbalance ⚠️)
> One or more L1 Slices have a much higher number of active cycles than the average. Maximum is **29.11%** above average.

> OPT - **Est. Speedup: 5.336%** (L2 load imbalance)
> One or more L2 Slices have a much higher number of active cycles than the average. Maximum is 13.04% above average.

---

## Summary & Findings

### 🎯 Winner: **Br=32**
- **17.2% faster execution** (10.33ms vs 12.10ms)
- **18.8% higher memory throughput** (56.27% vs 47.32%)
- **168% higher L2 cache throughput** (7.80% vs 2.91%)

### Critical Observations

| Issue | Br=32 | Br=64 |
|---|---|---|
| **Resource Bottleneck** | Registers (limit: 4 blocks) | **Registers + SRAM** (limit: 2 blocks) ⚠️ |
| **Shared Memory Pressure** | 19.01 KB/block | **37.12 KB/block** (95% more) ⚠️ |
| **Load Imbalance Potential** | 13.62% max speedup | **30.40% max speedup** ⚠️ |
| **L2 Cache Utilization** | ✅ Good (7.80%) | ❌ Poor (2.91%) |
| **Achieved Occupancy** | 53.78% | 57.83% (higher but starved) |

### Why Br=32 Wins Despite Lower Occupancy

1. **Better L2 Cache Efficiency** - 2.6x more L2 activity indicates more effective buffer reuse
2. **More Kernel Launches** - 2x grid size (256 vs 128 blocks) enables better GPU scheduling
3. **Less Resource Contention** - Fewer blocks resident per SM reduces register/SRAM pressure
4. **Superior Throughput** - Despite lower occupancy, achieves better realized performance

### Why Br=64 Struggles

1. **Dual Resource Constraints** - Limited by **both** registers **and** shared memory
2. **More Severe Load Imbalance** - 29.11% vs 15.29% max deviation across SMs
3. **Poor L2 Utilization** - Only 2.91% throughput (vs 7.80% for Br=32)
4. **Shared Memory Starving Occupancy** - Only 2 blocks resident vs 5 for Br=32

### Recommendations

✅ **Use Br=32**. Although Br=64 has higher theoretical speedup potential (33.33% vs 19.32%), Br=32 better utilizes the memory hierarchy and demonstrates superior real-world performance. The smaller blocks enable more effective scheduling and significantly better L2 cache behavior.




