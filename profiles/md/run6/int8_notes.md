# Quantization Strategy

## I. Symmetric vs Asymmetric Quantization (int8)

### I.1. Single Array Quantization/Dequantization

#### I.1.1. **Symmetric** (Recommended for matmuls)
*Explicitly centers around 0*

**Formulas:**
```
scale = max(|min|, |max|) / 127
zp = 0
```

**Quantize:** `q = clamp(round(x / scale), -128, 127)`

**Dequantize:** `x ≈ q * scale`

---

#### I.1.2. **Asymmetric** (Better for non-negative/skewed ranges)
*Calculates where true zero falls in int8 space*

Example: fp32 range `[-10, 100]` → 0 maps to int8 value ≈ -105 (this is `zp`)

**Formulas:**
```
scale = (max - min) / 255
zp = round(-128 - min / scale)  // int8 representation of float 0
```

**Quantize:** `q = clamp(round((x - min) / scale), -128, 127)`

**Dequantize:** `x ≈ (q - zp) * scale`

---

### I.2. Matmul `A@B` Quantization/Dequantization

**Input:** A, B given in fp32  
**Hardware:** WMMA int8 (int8 @ int8 → int32 only)

#### I.2.1. Quantization Phase
- Compute: `scale_A, zp_A, A_int8`
- Compute: `scale_B, zp_B, B_int8`

#### I.2.2. WMMA Computation
```
res_int32 = A_int8 @ B_int8
```

#### I.2.3. Dequantization Phase

**Symmetric (Simpler, Recommended):**
```
res_fp32 = res_int32 * scale_A * scale_B
```

**Asymmetric (More Precise, Higher Overhead):**
```
res_fp32 = res_int32 * scale_A * scale_B 
         - (rowsum_A * zp_B + colsum_B * zp_A - K * zp_A * zp_B) * scale_A * scale_B
```

where:
- `rowsum_A[i]` = sum of row i in `A_int8` (M values)
- `colsum_B[j]` = sum of column j in `B_int8` (N values)
- `K` = inner dimension of matmul

> **⚠️ Note:** Asymmetric requires computing row/column sums of (potentially massive) matrices → production uses symmetric unless precision is critical.

---

## II. MHA Flash Attention with Int8 Quantization (High-Level Overview)

### General Quantization Strategy

**Goal:** Reduce compute/memory by quantizing matrix multiplications while maintaining numerical stability.

**Key Operations:**
1. **Input quantization:** Q, K, V → int8 (per-block symmetric with separate scales)
2. **Matmul:** int8 @ int8 → int32 (via WMMA hardware)
3. **Dequantization:** int32 → fp32 with product of input scales
4. **Softmax:** Computed in fp32 (requires floating-point exponentials and comparisons)
5. **Softmax quantization:** Softmax probabilities (weights) → int8 (before matmul with V)
6. **Output matmul & accumulation:** int8 @ int8 → int32, dequantize to fp32, accumulate

**Quantization Points in Pipeline:**
- `Q @` `K^T`: `Q_int8 @ K_int8^T → scores_int32 → dequant to fp32`
- Softmax: `fp32 exp/max/sum operations` (no quantization)
- `P @ V`: `P_int8 @ V_int8 → output_partial_int32 → dequant to fp32 → accumulate`

### Key Design Decisions
✅ Use **symmetric quantization** for all matmuls (centered around 0, no zero-point)  
✅ Keep softmax in **float32** (no quantization)  
✅ Separate `scale` for each block: Q, K, V, P (different ranges per block)  
✅ Dequantize immediately after each matmul requiring the result in fp32

---

---

## III. Online Flash Attention with Int8 Quantization (Detailed Algorithm)

This section describes the actual implementation: block-wise processing with online softmax statistics tracking.

### Algorithm Structure (per query block)

**Initialization (before kv_block loop):**
- Load & quantize Q block (Br × d) → int8
- Initialize `output = 0` (Br × d, **fp32 accumulator**)
- Initialize `stats`: sum_exp=0, max_prev=0 (dummy for first iteration) per row

**Per kv_block_idx iteration (process one K,V block):**
1. Load K & V blocks (Bc × d each), quantize → int8
2. Compute `scores_int32 = Q_int8 @ K_int8^T` (Br × Bc)
3. **Online Softmax:**
   - Dequantize scores → fp32: `scores_fp32 = scores_int32 * scale_Q * scale_K`
   - Find max per row, scale by `1/sqrt(d)`
   - Compute `exp(score - max)`, accumulate sum across warp
   - Update stats: `sum_exp = exp(max_old - max_new) * sum_exp_old + sum_new` (handles numerical stability across blocks)
   - Rescale previous output by `exp(max_old - max_new)` (maintains consistency with new max)
4. **Quantize softmax:** `P_int8 = quantize(scores_fp32)` (separate `scale_P`)
5. **Accumulate output:**
   - Compute `partial_int32 = P_int8 @ V_int8` (Br × d)
   - **Immediately dequantize to fp32:** `partial_fp32 = partial_int32 * scale_P * scale_V`
   - Add to output: `output += partial_fp32`
6. Copy `max_curr → max_prev` for next iteration

**Epilogue (after kv_block loop):**
- Normalize: `output_final = output / sum_exp` (both already fp32)
- Store to global memory

### Key Insight
Each P_i @ V_i iteration has **distinct scales** (scale_P_i and scale_V_i), so dequantization must happen immediately after each matmul. Output is accumulated in fp32 to avoid mixing heterogeneous scaled int32 values.


---

## IV. Implementation Notes & Optimization Challenges

### SRAM Optimization & Memory Aliasing

**Challenge:** Int8 quantization requires additional buffers for int8 and int32 data, increasing shared-memory pressure.

**Solution:** Implemented shared-memory union (aliasing) for temporary buffers:
- `scores_fp32`, `scores_int32`, `temp_output_int32`
- `q_block`, `scores_int8` (used for online softmax accumulation)

**Trade-off:** Code readability decreased; debugging complexity increased.

**Critical Constraint:** Do not modify WMMA input/output buffers — maintaining 16-byte alignment is essential.

---

### SRAM Budget Bottleneck

**Primary issue:** The `c_scratch` buffer in WMMA, required to accumulate partial results from left and right warps per tile row, consumes the largest portion of shared memory.

**Unexpected finding:** SRAM configuration size varies significantly:
- **Br=32:** 102.4 kB
- **Br=64:** 65.54 kB (60% less)

This causes Br=64 to fit only 1 block/SM instead of 3, **reducing latency by 40%** (10 ms → 14 ms).

**Solution:** Use `cudaFuncSetAttribute()` in the launcher to force maximum SRAM configuration and maintain 102.4 kB for both settings.

---

### Register Pressure Optimization Attempts (Br=32)

At Br=32, block-limit bottleneck is **registers** (limit: 3 blocks). Attempted solutions:

| Approach | Register Savings | Throughput Impact | Latency Impact | Result |
|---|---|---|---|---|
| Replace `__forceinline__` with `__noinline__` | ~6–7% | **−4–5%** | **+10%** | ❌ Counterproductive |
| Move stats arrays to shared memory | N/A | Neutral | Neutral | ❌ No improvement |
| Remove `#pragma unroll` | N/A | Negative | Negative | ❌ No improvement |

**Key learning:** `-maxrregcount=X` applies **compiler-wide optimization**, not just capping per-thread regs. Example: `-maxrregcount=68` reduced per-thread usage from 69 → 60 regs.

---

### Strategic Pivot: Multiple Smaller Blocks Over Fewer Large Blocks

**Original goal:** Maximize warps per block to improve occupancy.

**New goal:** Fit more smaller blocks per SM for better scheduling and load balancing (at the cost of fewer warps/block).

**Outcome:** Achieved better occupancy while reducing per-block resource overhead and improving SM utilization.

---

### Shared Memory Configuration Bug & Fix

**Issue:** Adding `temp_output_int32` to the shared union unexpectedly dropped SRAM config from 102.4 kB → 65.54 kB, halving block residency.

**Root cause:** CUDA runtime dynamically sets SRAM configuration based on detected shared-memory usage patterns.

**Fix:** Invoke ` cudaFuncSetAttribute((void*)kernel<...>, cudaFuncAttributePreferredSharedMemoryCarveout, 100)` in the launcher to lock SRAM at `100%` configuration, preserving block residency and latency.