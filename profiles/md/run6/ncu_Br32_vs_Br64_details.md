# Nsight Compute - Detailed Analysis

Kernels profiled: [fa_tc_int8_a.cu](../../../mha_kernels/fa_tc_int8_a.cu) in two settings: Br = 32 and Br = 64.

## Goal



Results: 


**Kernels:**
- **fa_tc_int8_a:**  [fa_tc_int8_a.cu](../../../mha_kernels/fa_tc_int8_a.cu), PAD = 0

We use TC WMMA tile size `8×32×16` with `8 × 2 = 16` warps (Br = 64) or `4 × 2 = 8` warps (Br = 32) working independently over the `Br × d` chunk of Q. The `d` dimension is split in two halves, distributed one per warp tile row.

### Result



## Profiling results

### Bottlenecks


![Bottlenecks](../../images/run6/int8a_bottlenecks.png)


## Comparative Analysis

### Compute vs Memory Throughput

![Compute and Memory Throughput](../../images/run6/int8a_throughput.png)

### Compute Workload


![Compute Workload](../../images/run6/int8a_compute_workload.png)

### Memory Workload and Bank Conflicts
 

![Bank Conflicts](../../images/run6/int8a_mem_workload.png)

### Scheduler Statistics


![Scheduler Statistics](../../images/run6/int8a_scheduler_stats.png)

### Warp State Statistics


![Warp State Statistics](../../images/run6/int8a_warp_state_stats.png)

### Occupancy

**Occupancy metrics:**

![Occupancy 1](../../images/run6/int8a_warp_launch_stats_and_occupancy.png)

![Occupancy 2](../../images/run6/int8a_warp_launch_stats_and_occupancy.png)

### Source Counters

![Source Counters](../../images/run6/int8a_warp_launch_stats_and_occupancy.png)

### Source Code Analysis

![Source Code 1](../../images/run6/int8a_source_code_regs_spill.png)

![Source Code 2](../../images/run6/int8a_source_code_regs_spill_2.png)

![Source Code 3](../../images/run6/int8a_source_code_uncoalesced_sram_write.png)

## Notes


## Next Steps
