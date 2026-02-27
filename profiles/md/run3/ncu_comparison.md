# NCU Performance Comparison

**File 1:** `fa.txt` - Original Flash Attention  
**File 2:** `fa_tc_v1.txt` - Flash Attention with Tensor Cores

---

## GPU Speed Of Light Throughput

| Metric | Unit | fa | fa_tc_v1 |
|---|---|---|---|
| DRAM Frequency | Ghz | 6.24 | 6.24 |
| SM Frequency | Mhz | 794.99 | 794.98 |
| Elapsed Cycles | cycle | 6623369 | 4585386 |
| Memory Throughput | % | 67.83 | 46.05 |
| DRAM Throughput | % | 0.15 | 0.21 |
| Duration | ms | 8.33 | 5.77 |
| L1/TEX Cache Throughput | % | 91.24 | 67.52 |
| L2 Cache Throughput | % | 2.55 | 3.59 |
| SM Active Cycles | cycle | 4917215.60 | 3123328.93 |
| Compute (SM) Throughput | % | 67.83 | 46.05 |

**Notes:**
- fa: INF - Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced.
- fa_tc_v1: OPT - This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance.

---

## Launch Statistics

| Metric | Unit | fa | fa_tc_v1 |
|---|---|---|---|
| Block Size |  | 512 | 128 |
| Function Cache Configuration |  | CachePreferNone | CachePreferNone |
| Grid Size |  | 128 | 128 |
| Registers Per Thread | register/thread | 54 | 80 |
| Shared Memory Configuration Size | Kbyte | 65.54 | 102.40 |
| Driver Shared Memory Per Block | Kbyte/block | 1.02 | 1.02 |
| Dynamic Shared Memory Per Block | Kbyte/block | 29.82 | 0 |
| Static Shared Memory Per Block | byte/block | 0 | 43.78 |
| # SMs | SM | 58 | 58 |
| Threads | thread | 65536 | 16384 |
| Uses Green Context |  | 0 | 0 |
| Waves Per SM |  | 1.10 | 1.10 |

---

## Occupancy

| Metric | Unit | fa | fa_tc_v1 |
|---|---|---|---|
| Block Limit SM | block | 24 | 24 |
| Block Limit Registers | block | 2 | 6 |
| Block Limit Shared Mem | block | 2 | 2 |
| Block Limit Warps | block | 3 | 12 |
| Theoretical Active Warps per SM | warp | 32 | 8 |
| Theoretical Occupancy | % | 66.67 | 16.67 |
| Achieved Occupancy | % | 54.13 | 15.22 |
| Achieved Active Warps Per SM | warp | 25.98 | 7.31 |

**Notes:**
- fa_tc_v1: OPT - Est. Local Speedup: 83.33%. The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the required amount of shared memory.

---

## GPU and Memory Workload Distribution

| Metric | Unit | fa | fa_tc_v1 |
|---|---|---|---|
| Average DRAM Active Cycles | cycle | — | 76426.67 |
| Total DRAM Elapsed Cycles | cycle | — | 216112128 |
| Average L1 Active Cycles | cycle | — | 3123328.93 |
| Total L1 Elapsed Cycles | cycle | — | 265597016 |
| Average L2 Active Cycles | cycle | — | 903918.92 |
| Total L2 Elapsed Cycles | cycle | — | 114202008 |
| Average SM Active Cycles | cycle | — | 3123328.93 |
| Total SM Elapsed Cycles | cycle | — | 265597016 |
| Average SMSP Active Cycles | cycle | — | 3123795.47 |
| Total SMSP Elapsed Cycles | cycle | — | 1062388064 |

**Notes:**
- fa_tc_v1: OPT - Est. Speedup: 21.65%. One or more SMs/SMSPs/L1 Slices have a much higher number of active cycles than the average.
