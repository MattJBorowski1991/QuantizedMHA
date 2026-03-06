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

## II. MHA Flash Attention with Int8 Quantization

### Workflow
1. **Input:** Q, K, V in fp32
2. **Quantize each:** Q→int8, K→int8, V→int8 (separate `scale` and `zp` for each)
3. **Compute scores:** `scores_int32 = Q_int8 @ K_int8^T` (WMMA)
4. **Dequantize scores:** `scores_fp32 = scores_int32 * scale_Q * scale_K`
5. **Apply scaling:** `scaled_scores = scores_fp32 / sqrt(d)` (float operation)
6. **Softmax:** `weights_fp32 = softmax(scaled_scores)` (float)
7. **Quantize weights:** `P_int8 = quantize(weights_fp32)` (separate `scale_P`, `zp_P`)
8. **Compute output:** `output_int32 = P_int8 @ V_int8` (WMMA)
9. **Dequantize output:** `output_fp32 = output_int32 * scale_P * scale_V`

---

### Key Design Decisions
✅ Use **symmetric quantization** for all matmuls (Q@K, P@V)  
✅ Keep softmax in **float32** (no quantization needed)  
✅ Separate `scale/zp` for Q, K, V, P (different ranges)
