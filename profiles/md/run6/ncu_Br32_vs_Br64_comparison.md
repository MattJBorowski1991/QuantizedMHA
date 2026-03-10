# Br=32 vs Br=64 Profiling Comparison

High-level NCU comparison of two fa_tc_v3 configurations: **Br=64 with 8×2=16 warps/block** vs **Br=32 with 4×2=8 warps/block**, with warp work distributed across both the block row dimension (Br) and embedding dimension (d). Both kernels compiled with `-maxrregcount=40` to control register pressure.

> **Compilation Flag**: Both kernels compiled with `-maxrregcount=40` to limit register pressure and allow more blocks resident on GPU.

## GPU Speed Of Light Throughput

| Metric Name | Br=32 | Br=64 |
|---|---|---|
| **DRAM Frequency (Ghz)** | 6.24 | 6.24 |
| **SM Frequency (Mhz)** | 794.99 | 795.36 |
| **Elapsed Cycles** | 7,190,366 | 9,707,774 |
| **Memory Throughput (%)** | **68.95** | 50.18 |
| **DRAM Throughput (%)** | 0.23 | 0.14 |
| **Duration (ms)** | **9.04** | 12.18 🎯 |
| **L1/TEX Cache Throughput (%)** | 77.68 | 71.87 |
| **L2 Cache Throughput (%)** | **5.69** | 2.71 ⚠️ |
| **SM Active Cycles** | 6,388,004.91 | 6,770,460.12 |
| **Compute (SM) Throughput (%)** | 68.95 | 50.18 ⚠️ |

### Nsight Compute Observations

**Br=32:**
> INF - Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the Compute Workload Analysis and Memory Workload Analysis sections.

**Br=64:**
> OPT - This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

---

## Launch Statistics

| Metric Name | Br=32 | Br=64 |
|---|---|---|
| **Block Size** | 256 | 512 |
| Function Cache Configuration | CachePreferNone | CachePreferNone |
| **Grid Size** | 256 | 128 |
| **Registers Per Thread** | 40 | 40 |
| Shared Memory Config Size (KB) | 102.40 | 102.40 |
| Driver Shared Memory Per Block (KB) | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block (B) | 0 | 0 |
| **Static Shared Memory Per Block (KB)** | 19.02 | **37.14** ⚠️ |
| # SMs | 58 | 58 |
| Threads | 65536 | 65536 |
| Uses Green Context | 0 | 0 |
| Waves Per SM | 0.88 | 1.10 |

### Key Observations

- **Br=64** uses nearly **2x static shared memory** (37.14 KB vs 19.02 KB)
- **Br=32** has **2x grid size** (256 vs 128) → better scheduling flexibility  
- **Br=32** with lower waves per SM but better throughput indicates more efficient execution

---

## Occupancy

| Metric Name | Br=32 | Br=64 |
|---|---|---|
| Block Limit SM | 24 | 24 |
| **Block Limit Registers** | 6 | **3** ⚠️ |
| **Block Limit Shared Mem** | 5 | **2** ⚠️ |
| **Block Limit Warps** | 6 | **3** ⚠️ |
| Theoretical Active Warps per SM | 40 | 32 |
| **Theoretical Occupancy (%)** | **83.33** | 66.67 |
| **Achieved Occupancy (%)** | 63.51 | **58.76** |
| **Achieved Active Warps Per SM** | 30.49 | **28.21** |

### Nsight Compute Observations

**Br=32:**
> OPT - **Est. Local Speedup: 23.79%**
> The difference between calculated theoretical (83.3%) and measured achieved occupancy (63.5%) can be the result of warp scheduling overheads or workload imbalances during the kernel execution.

**Br=64:**
> OPT - **Est. Local Speedup: 33.33%**
> The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the required amount of shared memory.

---

## GPU and Memory Workload Distribution

| Metric Name | Br=32 | Br=64 |
|---|---|---|
| **Average DRAM Active Cycles** | 129,701.33 | 107,074.67 |
| Total DRAM Elapsed Cycles | 338,882,560 | 456,486,912 |
| Average L1 Active Cycles | 6,388,004.91 | 6,770,460.12 |
| Total L1 Elapsed Cycles | 417,372,014 | 562,475,670 |
| **Average L2 Active Cycles** | **4,294,099.21** | 3,209,794.62 |
| Total L2 Elapsed Cycles | 179,080,632 | 241,227,000 |
| Average SM Active Cycles | 6,388,004.91 | 6,770,460.12 |
| Total SM Elapsed Cycles | 417,372,014 | 562,475,670 |
| Average SMSP Active Cycles | 6,300,193.55 | 6,656,480.79 |
| Total SMSP Elapsed Cycles | 1,669,488,056 | 2,249,902,680 |

### Nsight Compute Observations

**Br=32:**
> OPT - **Est. Speedup: 9.889%** (SM load imbalance)
> One or more SMs have a much higher number of active cycles than the average. Maximum is 11.14% above average.

> OPT - **Est. Speedup: 10.87%** (SMSP load imbalance)  
> One or more SMSPs have a much higher number of active cycles than the average. Additionally, other SMSPs have a much lower number of active cycles than the average. Maximum is 12.42% above average, minimum is 10.59% below average.

> OPT - **Est. Speedup: 9.889%** (L1 load imbalance)
> One or more L1 Slices have a much higher number of active cycles than the average. Maximum is 11.14% above average.

> OPT - **Est. Speedup: 7.572%** (L2 load imbalance)
> One or more L2 Slices have a much higher number of active cycles than the average. Maximum is 13.16% above average.

**Br=64:**
> OPT - **Est. Speedup: 21.12%** (SM load imbalance ⚠️)
> One or more SMs have a much higher number of active cycles than the average. Maximum is **30.26%** above average.

> OPT - **Est. Speedup: 21.55%** (SMSP load imbalance ⚠️)
> One or more SMSPs have a much higher number of active cycles than the average. Maximum is **31.40%** above average, minimum is 13.46% below average.

> OPT - **Est. Speedup: 21.12%** (L1 load imbalance ⚠️)
> One or more L1 Slices have a much higher number of active cycles than the average. Maximum is **30.26%** above average.

> OPT - **Est. Speedup: 6.495%** (L2 load imbalance)
> One or more L2 Slices have a much higher number of active cycles than the average. Maximum is 20.34% above average.

---

### Understanding the Load Imbalance Difference

The substantially larger SM/SMSP/L1 imbalance in Br=64 (30–31% vs 10–11% for Br=32) stems from **coarser block granularity**. With 58 SMs:
- **Br=32**: 256 blocks ÷ 58 SMs ≈ **4.4 blocks/SM** (fine-grained, smooth distribution)
- **Br=64**: 128 blocks ÷ 58 SMs ≈ **2.2 blocks/SM** (coarse-grained, high variance)

With only ~2 blocks per SM, some SMs inevitably get 3 blocks while others get 2, creating larger variance in compute activity. Within Br=64's larger blocks (8×2=16 warps), uneven work distribution across the Br dimension amplifies this imbalance. 

**Why L2 shows more balance in Br=32**: L2 is **globally shared**—it services all SMs uniformly regardless of per-SM block distribution. Br=32's higher L2 activity (5.69% vs 2.71%) indicates better global memory efficiency and stronger cache reuse across the kernel.

**Lesson**: More blocks → finer granularity → better dynamic load balancing.

---

## Summary & Findings

### 🎯 Winner: **Br=32**
- **34.8% faster execution** (9.04ms vs 12.18ms)
- **37.4% higher memory throughput** (68.95% vs 50.18%)
- **109.9% higher L2 cache throughput** (5.69% vs 2.71%)

### Critical Observations

| Issue | Br=32 | Br=64 |
|---|---|---|
| **Resource Bottleneck** | Registers (limit: 6 blocks) | **Registers + SRAM** (limit: 3 blocks) ⚠️ |
| **Shared Memory Pressure** | 19.02 KB/block | **37.14 KB/block** (95% more) ⚠️ |
| **Load Imbalance Potential** | 10.87% max speedup | **31.40% max speedup** ⚠️ |
| **L2 Cache Utilization** | ✅ Good (5.69%) | ❌ Poor (2.71%) |
| **Achieved Occupancy** | 63.51% | 58.76% (lower than Br=32) |

### Why Br=32 Wins Decisively

1. **Superior Compute Throughput** - 68.95% vs 50.18% indicates much better utilization
2. **Better L2 Cache Efficiency** - 2.1x more L2 activity demonstrates stronger memory reuse
3. **More Kernel Launches** - 2x grid size (256 vs 128 blocks) enables better GPU scheduling
4. **Higher Occupancy** - Theoretical 83.33% vs 66.67% gives better warp availability
5. **Well-Balanced Compute/Memory** - Nsight notes "well-balanced" vs "latency issues" for Br=64

### Why Br=64 Struggles

1. **Dual Resource Constraints** - Limited by **both** registers **and** shared memory (3 blocks max)
2. **Severe Load Imbalance** - 31.40% vs 10.87% max deviation across SMs
3. **Poor Energy Efficiency** - Low cache throughput (2.71%) suggests memory bottleneck
4. **Coarse Block Granularity** - Only 2.2 blocks/SM creates uneven work distribution

### Recommendations

✅ **Use Br=32**. It delivers **superior real-world performance** with 34.8% faster execution. While Br=64 has slightly less occupancy variance opportunity, Br=32's better memory hierarchy utilization, higher occupancy, and fine-grained scheduling make it the clear winner. With `-maxrregcount=40`, Br=32 achieves balanced compute and memory throughput—a hallmark of efficient kernel design.




