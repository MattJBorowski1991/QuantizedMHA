# NCU Performance Comparison: fa vs fa_tc_v1

## GPU Speed Of Light Throughput

| Metric Name | Metric Unit | fa | fa_tc_v1 |
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

**fa notes:**
- INF: Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the Compute Workload Analysis and Memory Workload Analysis sections.

**fa_tc_v1 notes:**
- OPT: This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.

## Launch Statistics

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

## Occupancy

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

**fa_tc_v1 notes:**
- OPT: Est. Local Speedup: 83.33% - The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the required amount of shared memory.

## GPU and Memory Workload Distribution

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


**fa_tc_v1 notes:**
- OPT: Est. Speedup: 21.65% - One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 31.74% above the average, while the minimum instance value is 12.44% below the average.
- OPT: Est. Speedup: 21.65% - One or more SMSPs have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 31.74% above the average, while the minimum instance value is 12.25% below the average.
- OPT: Est. Speedup: 21.65% - One or more L1 Slices have a much higher number of active cycles than the average number of active cycles. Maximum instance value is 31.74% above the average, while the minimum instance value is 12.44% below the average.
