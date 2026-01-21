# Quantized Flash-Attention with RoPE & JAX Integration

A high-performance CUDA library implementing a fused INT8/FP16 Multi-Head Attention kernel.

## Summary

- Core: Build GEMM kernels from scratch using WMMA (Tensor Cores) with manual tiling and double-buffering.
- Logic: Integrate Rotary Positional Embeddings (RoPE) directly into the Q/K loading phase to avoid global memory round-trips.
- Optimization: Use Online Softmax (FlashAttention approach) to minimize register pressure and shared memory bottlenecks.
- Profiling: Quantify performance using Nsight Compute, specifically targeting Compute Throughput (SM %) and Memory Bandwidth utilization.
- Integration: Expose via Pybind11 as a JAX Custom Call, enabling use within XLA-compiled pipelines.

## Proposed Extra Features (Choose 1-2)

- **Grouped-Query Attention (GQA):** The current industry standard for efficient inference (used in Llama 3).
- **Causal Masking Fusion:** Fuse the look-ahead mask directly into the attention score calculation to save bandwidth.
- **KV-Cache Support:** Implement logic to handle incremental decoding for LLM inference.
- **Sliding Window Attention (SWA):** Optimize for long-context sequences by limiting attention scope.
- **Stochastic Rounding:** A sophisticated way to handle precision loss during INT8 quantization.

## Crucial Amendments

- **Unit Tests vs. Reference:** Include a Python test suite that compares your kernel outputs against `jax.nn.attention` to prove correctness and quantify the Floating Point Error (L2 Norm).
- **Benchmarking:** Create a README.md with a table comparing your kernel's TFLOPS and Latency against a naive implementation and a standard library (like cuDNN or FlashAttention 2).
- **Memory Pipelining:** Use `cp.async` (Async Copy) if targeting Ampere (RTX 30-series) or newer to overlap data movement with computation.
- **Bank Conflicts:** Explicitly document how you avoided shared memory bank conflicts in your code comments; this is a common interview "trap" question.





***** KERNEL COMPONENTS **********

1. **Quantized GEMM (INT8)**
	1.1 **Tiled:** Manual Shared Memory caching to minimize global memory bandwidth.
	1.2 **WMMA + Async:** Using Tensor Cores via `mma.sync` or `wmma`, utilizing `cp.async` (for Ampere+) and double-buffering to hide latency.
2. **Rotary Positional Embeddings (RoPE)** - apply the rotation math (sine/cosine) inside the MHA kernel to Q and K as they are being loaded from DRAM to SRAM/registers.
3. **Multi-Head Attention (MHA)**
	3.1 **Unfused (Baseline):** Three separate kernels (Q*K^T) -> softmax -> *V to quantify the speedup of fusion.
	3.2 **Flash-Logic (The Goal):** Implement the Online Softmax algorithm. This allows you to compute attention on arbitrary sequence lengths without running out of shared memory by updating the softmax normalization factor incrementally.

4. **System Integration**
	- **JAX Custom Call:** Using Pybind11 to register these kernels, allowing you to run `jax.jit(my_cuda_attention)(...)` for end-to-end testing.