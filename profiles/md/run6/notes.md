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
