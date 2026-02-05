# Nsight Compute Profiling Comparison

Detailed performance metrics comparing unfused attention components vs fused FA_4X4 implementation.

---

## GPU Speed Of Light Throughput

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
DRAM Frequency (Ghz)            | 6.24                  | 6.24                  | 6.24                 | 6.24
SM Frequency (Mhz)              | 802.42                | 796.06                | 802.53               | 803.96
Elapsed Cycles (cycle)          | 2553348               | 385440                | 2289480              | 7389404
Memory Throughput (%)           | 82.50                 | 87.59                 | 89.63                | 45.88
DRAM Throughput (%)             | 6.34                  | 87.59                 | 12.35                | 0.14
Duration                        | 3.14 ms               | 482.50 us             | 2.82 ms              | 9.07 ms
L1/TEX Cache Throughput (%)     | 82.61                 | 49.09                 | 92.01                | 76.87
L2 Cache Throughput (%)         | 13.47                 | 57.95                 | 14.91                | 1.19
SM Active Cycles (cycle)        | 2516810.64            | 378633.47             | 2201512.84           | 4353733.55
Compute (SM) Throughput (%)     | 82.50                 | 30.81                 | 89.63                | 45.88


      OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.6 full
            waves across all SMs. Look at Launch Statistics for more details.

---

## PM Sampling

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Maximum Buffer Size (Mbyte)     | 50.33                 | 25.17                 | 50.33                | 28.84
Dropped Samples (sample)        | 0                     | 0                     | 0                    | 0
Maximum Sampling Interval (us)  | 1                     | 1                     | 1                    | 4
# Pass Groups                   | 2                     | 2                     | 2                    | 2

---

## Compute Workload Analysis

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Executed Ipc Active (inst/c)    | 1.46                  | 0.97                  | 1.56                 | 2.02
Executed Ipc Elapsed (inst/c)   | 1.46                  | 0.95                  | 1.52                 | 1.20
Issue Slots Busy (%)            | 36.55                 | 24.19                 | 39.06                | 50.49
Issued Ipc Active (inst/c)      | 1.46                  | 0.97                  | 1.56                 | 2.02
SM Busy (%)                     | 36.55                 | 24.19                 | 39.06                | 50.49


      INF   FMA is the highest-utilized pipeline (27.4%) based on active cycles, taking into account the rates of its
            different instructions. It executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD)
            operations. It is well-utilized, but should not be a bottleneck.

---

## Memory Workload Analysis

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Memory Throughput               | 19.00 Gbyte/s         | 262.49 Gbyte/s        | 37.00 Gbyte/s        | 416.51 Mbyte/s
Mem Busy (%)                    | 65.98                 | 53.93                 | 49.22                | 30.78
Max Bandwidth (%)               | 82.50                 | 87.59                 | 89.63                | 45.88
L1/TEX Hit Rate (%)             | 71.46                 | 21.03                 | 1.12                 | 0.05
L2 Compression Success (%)      | 0                     | 0                     | 0                    | 0
L2 Hit Rate (%)                 | 99.49                 | 79.62                 | 87.18                | 97.65
Mem Pipes Busy (%)              | 82.50                 | 30.81                 | 89.63                | 45.88


      OPT   Est. Speedup: 11.76%
            The memory access pattern for shared loads might not be optimal and causes on average a 1.2 - way bank
            conflict across all 81289216 shared load requests. This results in 14680064 bank conflicts, which represent
            15.30% of the overall 95969280 wavefronts for shared loads. Check the Source Counters section for
            uncoalesced shared loads.
      
      OPT   Est. Speedup: 52.71%
            The memory access pattern for shared stores might not be optimal and causes on average a 3.2 - way bank
            conflict across all 8413440 shared store requests. This results in 18351335 bank conflicts, which represent
            68.56% of the overall 26764943 wavefronts for shared stores. Check the Source Counters section for
            uncoalesced shared stores.

---

## Scheduler Statistics

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
One or More Eligible (%)        | 36.55                 | 24.19                 | 39.07                | 50.48
Issued Warp Per Scheduler       | 0.37                  | 0.24                  | 0.39                 | 0.50
No Eligible (%)                 | 63.45                 | 75.81                 | 60.93                | 49.52
Active Warps Per Scheduler      | 11.76                 | 11.39                 | 10.55                | 4.46
Eligible Warps Per Scheduler    | 1.56                  | 0.33                  | 1.57                 | 1.36


      OPT   Est. Local Speedup: 49.52%
            Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only
            issues an instruction every 2.0 cycles. This might leave hardware resources underutilized and may lead to
            less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average
            of 4.46 active warps per scheduler, but only an average of 1.36 warps were eligible per cycle.

---

## Warp State Statistics

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Warp Cycles Per Issued Instr    | 32.18                 | 47.09                 | 27.00                | 8.83
Warp Cycles Per Executed Instr  | 32.18                 | 47.14                 | 27.00                | 8.84
Avg. Active Threads Per Warp    | 32                    | 32                    | 32                   | 16.04
Avg. Not Predicated Off Threads | 30.19                 | 30.64                 | 30.31                | 14.91


      OPT   Est. Speedup: 24.5%
            Instructions are executed in warps, which are groups of 32 threads. Optimal instruction throughput is
            achieved if all 32 threads of a warp execute the same instruction. The chosen launch configuration, early
            thread completion, and divergent flow control can significantly lower the number of active threads in a warp
            per cycle. This workload achieves an average of 16.0 threads being active per cycle.

---

## Instruction Statistics

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Avg. Executed Instr Per Sched   | 919763.86             | 91489.10              | 859665.66            | 2195377.66
Executed Instructions           | 213385216             | 21225472              | 199442432            | 509327616
Avg. Issued Instr Per Sched     | 919898.53             | 91583.81              | 859800.91            | 2198257.56
Issued Instructions             | 213416459             | 21247443              | 199473810            | 509995755


    (No specific OPT/INF for Instruction Statistics)

---

## Launch Statistics

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Block Size                      | 256                   | 256                   | 256                  | 512
Function Cache Configuration    | CachePreferNone       | CachePreferNone       | CachePreferNone      | CachePreferNone
Grid Size                       | 65536                 | 4096                  | 1024                 | 64
Registers Per Thread            | 37                    | 39                    | 37                   | 64
Shared Memory Config Size (KB)  | 65.54                 | 32.77                 | 65.54                | 102.40
Driver Shared Mem Per Block     | 1.02                  | 1.02                  | 1.02                 | 1.02
Dynamic Shared Mem Per Block    | 0                     | 0                     | 0                    | 33.54
Static Shared Mem Per Block     | 2.05                  | 1.02                  | 2.05                 | 0
# SMs                           | 58                    | 58                    | 58                   | 58
Stack Size                      | 1024                  | 1024                  | 1024                 | 1024
Threads                         | 16777216              | 1048576               | 262144               | 32768
# TPCs                          | 29                    | 29                    | 29                   | 29
Uses Green Context              | 0                     | 0                     | 0                    | 0
Waves Per SM                    | 188.32                | 11.77                 | 2.94                 | 0.55


      OPT   If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two
            blocks per multiprocessor (compared to the currently executed 1.1 blocks) This way, blocks that aren't
            waiting for __syncthreads() can keep the hardware busy.

---

## Occupancy

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Block Limit SM                  | 24                    | 24                    | 24                   | 24
Block Limit Registers           | 6                     | 6                     | 6                    | 2
Block Limit Shared Mem          | 21                    | 16                    | 21                   | 2
Block Limit Warps               | 6                     | 6                     | 6                    | 3
Theoretical Active Warps Per SM | 48                    | 48                    | 48                   | 32
Theoretical Occupancy (%)       | 100                   | 100                   | 100                  | 66.67
Achieved Occupancy (%)          | 98.16                 | 95.47                 | 87.87                | 37.13
Achieved Active Warps Per SM    | 47.12                 | 45.83                 | 42.18                | 17.82


      OPT   Est. Speedup: 44.3%
            The difference between calculated theoretical (66.7%) and measured achieved occupancy (37.1%) can be the
            result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can
            occur between warps within a block as well as across blocks of the same kernel.
      
      OPT   Est. Speedup: 33.33%
            The 8.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the
            hardware maximum of 12. This kernel's theoretical occupancy (66.7%) is limited by the number of required
            registers, and the required amount of shared memory.

---

## GPU and Memory Workload Distribution

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Average DRAM Active Cycles      | 1243242.67            | 2638565.33            | 2171085.33           | 78736
Total DRAM Elapsed Cycles       | 117672960             | 18074624              | 105517056            | 339979264
Average L1 Active Cycles        | 2516810.64            | 378633.47             | 2201512.84           | 4353733.55
Total L1 Elapsed Cycles         | 146165764             | 22278102              | 131080798            | 423096870
Average L2 Active Cycles        | 2381070.71            | 391517.04             | 2267760.58           | 356106.33
Total L2 Elapsed Cycles         | 62184768              | 9551880               | 55759536             | 179659944
Average SM Active Cycles        | 2516810.64            | 378633.47             | 2201512.84           | 4353733.55
Total SM Elapsed Cycles         | 146165764             | 22278102              | 131080798            | 423096870
Average SMSP Active Cycles      | 2517000.12            | 378659.44             | 2200947.55           | 4354909.20
Total SMSP Elapsed Cycles       | 584663056             | 89112408              | 524323192            | 1692387480

  FA_4X4:
    fa_kernel:
      OPT   Est. Speedup: 23.86%
            One or more SMs have a much higher number of active cycles than the average number of active cycles. Maximum
            instance value is 39.99% above the average, while the minimum instance value is 8.27% below the average.

      OPT   Est. Speedup: 23.83%
            One or more SMSPs have a much higher number of active cycles than the average number of active cycles.
            Maximum instance value is 39.92% above the average, while the minimum instance value is 8.22% below the
            average.

      OPT   Est. Speedup: 23.86%
            One or more L1 Slices have a much higher number of active cycles than the average number of active cycles.
            Maximum instance value is 39.99% above the average, while the minimum instance value is 8.27% below the
            average.

---

## Source Counters

                                | Unfused                                                              | FA_4X4
Metric Name                     | mma_A_Bt              | softmax               | mma_A_B              | fa_kernel
--------------------------------|-----------------------|-----------------------|----------------------|-----------------------
Branch Instructions Ratio (%)   | 0.10                  | 0.10                  | 0.09                 | 0.05
Branch Instructions             | 20447232              | 2146304               | 18898944             | 26462720
Branch Efficiency (%)           | 100                   | 100                   | 100                  | 98.67
Avg. Divergent Branches         | 0                     | 0                     | 0                    | 1129.93

  FA_4X4:
    fa_kernel:
      OPT   Est. Speedup: 16.06%
            This kernel has uncoalesced shared accesses resulting in a total of 33030144 excessive wavefronts (27% of the
            total 122732800 wavefronts). Check the L1 Wavefronts Shared Excessive table for the primary source
            locations. The CUDA Best Practices Guide has an example on optimizing shared memory accesses.

---

**End of Profiling Comparison**
