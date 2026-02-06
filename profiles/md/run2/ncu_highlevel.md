# Nsight Compute Profiling Comparison (run2)

Detailed performance metrics comparing unfused attention components vs fused FA_4X4 implementation (run2).

---

## GPU Speed Of Light Throughput

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| DRAM Frequency (Ghz) | 6.24 | 6.24 | 6.24 | 6.24 |
| SM Frequency (Mhz) | 840.40 | 858.91 | 851.17 | 830.81 |
| Elapsed Cycles (cycle) | 5182838 | 1898344 | 4528810 | 6549900 |
| Memory Throughput (%) | 82.63 | 89.61 | 89.79 | 68.35 |
| DRAM Throughput (%) | 16.43 | 89.61 | 25.92 | 0.15 |
| Duration | 6.14 ms | 2.20 ms | 5.30 ms | 7.85 ms |
| L1/TEX Cache Throughput (%) | 82.68 | 40.27 | 91.78 | 91.96 |
| L2 Cache Throughput (%) | 14.40 | 48.35 | 15.22 | 2.66 |
| SM Active Cycles (cycle) | 5160453.86 | 1882557.21 | 4413095.83 | 4848924.40 |
| Compute (SM) Throughput (%) | 82.63 | 22.08 | 89.79 | 68.35 |


      INF   The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device.

---

## PM Sampling

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Maximum Buffer Size (Mbyte) | 23.53 | 33.55 | 20.32 | 30.02 |
| Dropped Samples (sample) | 91 | 0 | 98 | 36 |
| Maximum Sampling Interval (us) | 2 | 32 | 2 | 256 |
| # Pass Groups | 2 | 2 | 2 | 2 |

---

## Compute Workload Analysis

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Executed Ipc Active (inst/c) | 1.52 | 0.69 | 1.56 | 1.45 |
| Executed Ipc Elapsed (inst/c) | 1.52 | 0.69 | 1.52 | 1.08 |
| Issue Slots Busy (%) | 38.01 | 17.36 | 38.94 | 36.28 |
| Issued Ipc Active (inst/c) | 1.52 | 0.69 | 1.56 | 1.45 |
| SM Busy (%) | 38.01 | 17.36 | 38.94 | 36.28 |


      INF   ALU / compute utilization notes vary per kernel; see individual sections for details.

---

## Memory Workload Analysis

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Memory Throughput | 49.24 Gbyte/s | 268.60 Gbyte/s | 77.69 Gbyte/s | 456.46 Mbyte/s |
| Mem Busy (%) | 66.36 | 45.63 | 49.31 | 43.73 |
| Max Bandwidth (%) | 82.63 | 89.61 | 89.79 | 68.35 |
| L1/TEX Hit Rate (%) | 70.70 | 19.39 | 0.64 | 0.40 |
| L2 Compression Success (%) | 0 | 0 | 0 | 0 |
| L2 Hit Rate (%) | 99.75 | 80.00 | 74.45 | 98.84 |
| Mem Pipes Busy (%) | 82.63 | 22.08 | 89.79 | 68.35 |


      OPT   Est. Speedups and memory access recommendations are shown in the per-kernel analysis above.

---

## Scheduler Statistics

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| One or More Eligible (%) | 38.02 | 17.41 | 38.95 | 36.28 |
| Issued Warp Per Scheduler | 0.38 | 0.17 | 0.39 | 0.36 |
| No Eligible (%) | 61.98 | 82.59 | 61.05 | 63.72 |
| Active Warps Per Scheduler | 11.57 | 11.59 | 10.57 | 3.72 |
| Eligible Warps Per Scheduler | 1.61 | 0.22 | 1.54 | 0.66 |


      OPT   Local speedup estimates and scheduler observations are included in kernel sections.

---

## Warp State Statistics

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Warp Cycles Per Issued Instr | 30.43 | 66.55 | 27.15 | 10.26 |
| Warp Cycles Per Executed Instr | 30.43 | 66.57 | 27.15 | 10.26 |
| Avg. Active Threads Per Warp | 32 | 32 | 32 | 32 |
| Avg. Not Predicated Off Threads | 30.08 | 31.24 | 30.31 | 30.81 |


      OPT   Warp-level stall breakdowns are in the full profiles; consider coalescing/shared-memory changes where noted.

---

## Instruction Statistics

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Avg. Executed Instr Per Sched | 1961560.28 | 326761.93 | 1718413.24 | 1757996.14 |
| Executed Instructions | 455081984 | 75808768 | 398671872 | 407855104 |
| Avg. Issued Instr Per Sched | 1961695.61 | 326856.64 | 1718548.23 | 1759159.47 |
| Issued Instructions | 455113382 | 75830740 | 398703190 | 408124996 |

---

## Launch Statistics

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Block Size | 256 | 256 | 256 | 256 |
| Function Cache Configuration | CachePreferNone | CachePreferNone | CachePreferNone | CachePreferNone |
| Grid Size | 262144 | 8192 | 1024 | 128 |
| Registers Per Thread | 37 | 39 | 37 | 54 |
| Shared Memory Config Size (KB) | 65.54 | 32.77 | 65.54 | 102.40 |
| Driver Shared Mem Per Block | 1.02 | 1.02 | 1.02 | 1.02 |
| Dynamic Shared Mem Per Block | 0 | 0 | 0 | 29.82 Kbyte/block |
| Static Shared Mem Per Block | 2.05 | 1.02 | 2.05 | 0 |
| # SMs | 58 | 58 | 58 | 58 |
| Threads | 67108864 | 2097152 | 262144 | 32768 |
| Uses Green Context | 0 | 0 | 0 | 0 |
| Waves Per SM | 753.29 | 23.54 | 2.94 | 0.74 |


      OPT   Launch configuration notes included in kernel sections (e.g., low waves per SM for `fa_kernel`).

---

## Occupancy

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Block Limit SM | 24 | 24 | 24 | 24 |
| Block Limit Registers | 6 | 6 | 6 | 4 |
| Block Limit Shared Mem | 21 | 16 | 21 | 3 |
| Block Limit Warps | 6 | 6 | 6 | 6 |
| Theoretical Active Warps Per SM | 48 | 48 | 48 | 24 |
| Theoretical Occupancy (%) | 100 | 100 | 100 | 50 |
| Achieved Occupancy (%) | 96.68 | 97.11 | 88.11 | 31.01 |
| Achieved Active Warps Per SM | 46.40 | 46.61 | 42.29 | 14.89 |


      OPT   Occupancy differences suggest FA kernel is limited by shared memory and register choices.

---

## GPU and Memory Workload Distribution

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Average DRAM Active Cycles | 6302016 | 12319170.67 | 8575805.33 | 74653.33 |
| Total DRAM Elapsed Cycles | 230162432 | 82484224 | 198531072 | 294135808 |
| Average L1 Active Cycles | 5160453.86 | 1882557.21 | 4413095.83 | 4848924.40 |
| Total L1 Elapsed Cycles | 299491052 | 109695502 | 261649762 | 378371076 |
| Average L2 Active Cycles | 4612062.38 | 1890935.54 | 4527558.67 | 1321096.54 |
| Total L2 Elapsed Cycles | 125477808 | 45954936 | 109655280 | 158448360 |
| Average SM Active Cycles | 5160453.86 | 1882557.21 | 4413095.83 | 4848924.40 |
| Total SM Elapsed Cycles | 299491052 | 109695502 | 261649762 | 378371076 |
| Average SMSP Active Cycles | 5160150.36 | 1876927.43 | 4412590.34 | 4848676.45 |
| Total SMSP Elapsed Cycles | 1197964208 | 438782008 | 1046599048 | 1513484304 |

+  FA_4X4:
+    fa_kernel:
+      OPT   Est. Speedup: 18.72%
+            One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum
+            instance value is 25.19% above the average, while the minimum instance value is 9.07% below the average.
+
+      OPT   Est. Speedup: 18.71%
+            One or more SMSPs have a much higher number of active cycles than the average number of active cycles.
+            Maximum instance value is 25.18% above the average, while the minimum instance value is 9.09% below the
+            average.
+
+      OPT   Est. Speedup: 18.72%
+            One or more L1 Slices have a much higher number of active cycles than the average number of active cycles.
+            Maximum instance value is 25.19% above the average, while the minimum instance value is 9.07% below the
+            average.
+
+---
+
+## Source Counters
+
+| | Unfused | | | FA_4X4 |
+|---|---|---|---|---|
+| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
+| Branch Instructions Ratio (%) | 0.10 | 0.09 | 0.09 | 0.05 |
+| Branch Instructions | 44040192 | 6979584 | 37773312 | 19756032 |
+| Branch Efficiency (%) | 100 | 100 | 100 | 100 |
+| Avg. Divergent Branches | 0 | 0 | 0 | 0 |
+
+  FA_4X4:
+    fa_kernel:
+      OPT   Est. Speedup: 31.65%
+            This kernel has uncoalesced shared accesses resulting in hotspot/excessive wavefronts as reported by the
+            profiler. Check the L1 Wavefronts Shared Excessive table and the Source page for primary source locations and
+            consider optimizing shared memory access patterns.
+
+---
+
+**End of Profiling Comparison (run2)**
+
+```
