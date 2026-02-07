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


    INF   Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. 
          Check both the Compute Workload Analysis and Memory Workload Analysis sections.    
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


    OPT   Est. Local Speedup: 83.1%                                                                                     
          All compute pipelines are under-utilized. Either this workload is very small or it doesn't issue enough warps 
          per scheduler. Check the Launch Statistics and Scheduler Statistics sections for further details.  
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


    OPT   Est. Local Speedup: 31.65%                                                                                    
          Every scheduler is capable of issuing one instruction per cycle, but for this kernel each scheduler only      
          issues an instruction every 2.8 cycles. This might leave hardware resources underutilized and may lead to     
          less optimal performance. Out of the maximum of 12 warps per scheduler, this kernel allocates an average of   
          3.72 active warps per scheduler, but only an average of 0.66 warps were eligible per cycle. Eligible warps    
          are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible   
          warp results in no instruction being issued and the issue slot remains unused. To increase the number of      
          eligible warps, reduce the time the active warps are stalled by inspecting the top stall reasons on the Warp  
          State Statistics and Source Counters sections.  
---

## Warp State Statistics

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Warp Cycles Per Issued Instr | 30.43 | 66.55 | 27.15 | 10.26 |
| Warp Cycles Per Executed Instr | 30.43 | 66.57 | 27.15 | 10.26 |
| Avg. Active Threads Per Warp | 32 | 32 | 32 | 32 |
| Avg. Not Predicated Off Threads | 30.08 | 31.24 | 30.31 | 30.81 |


    OPT   Est. Speedup: 31.65%                                                                                          
          On average, each warp of this kernel spends 3.7 cycles being stalled waiting for the MIO (memory              
          input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of  
          the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory        
          instructions. When caused by shared memory accesses, trying to use fewer but wider loads can reduce pipeline  
          pressure. This stall type represents about 36.4% of the total average of 10.3 cycles between issuing two      
          instructions.                                                                                                 
    ----- --------------------------------------------------------------------------------------------------------------
    INF   Check the Warp Stall Sampling (All Samples) table for the top stall locations in your source based on         
          sampling data. The Kernel Profiling Guide                                                                     
          (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference) provides more details    
          on each stall reason. 
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


    OPT   Est. Speedup: 31.65%                                                                                          
          The difference between calculated theoretical (50.0%) and measured achieved occupancy (31.0%) can be the      
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         
    ----- --------------------------------------------------------------------------------------------------------------
    OPT   Est. Speedup: 31.65%                                                                                          
          The 6.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the       
          hardware maximum of 12. This kernel's theoretical occupancy (50.0%) is limited by the required amount of      
          shared memory.                                                                                                

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


## Source Counters

| | Unfused | | | FA_4X4 |
|---|---|---|---|---|
| Metric Name | mma_A_Bt | softmax | mma_A_B | fa_kernel |
| Branch Instructions Ratio (%) | 0.10 | 0.09 | 0.09 | 0.05 |
| Branch Instructions | 44040192 | 6979584 | 37773312 | 19756032 |
| Branch Efficiency (%) | 100 | 100 | 100 | 100 |
| Avg. Divergent Branches | 0 | 0 | 0 | 0 |

  FA_4X4:
    fa_kernel:
      OPT   Est. Speedup: 31.65%
            This kernel has uncoalesced shared accesses resulting in hotspot/excessive wavefronts as reported by the
            profiler. Check the L1 Wavefronts Shared Excessive table and the Source page for primary source locations and
            consider optimizing shared memory access patterns.

---

**End of Profiling Comparison (run2)**
