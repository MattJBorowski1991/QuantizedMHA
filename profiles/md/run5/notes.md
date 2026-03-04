Goal: reduce bank conflicts from run4.


1. Padding

SRAM usage minimization so that PAD=16 no longer causes SRAM overflow:
1. Move c_scratch buffer from fp to half
2. Share the buffer for kt and values
3. Share the buffer for q_block and scores_fp16 (half)
4. Attempt sharing the buffer for scores and output (fp) in exact same way => bug with random incorrect kernel output values

Result: SRAM allocation reduced by over 20%.

After SRAM reduction, PAD=16 was made possible - it surprisingly turned out that PAD=8 removes more bank conflicts than PAD=16! And hence provided better perf of 6.2ms vs 6.8ms.


2. Swizzling

Tried various swizzling strategies, which did not yield perf improvement. The least bad one is implemented in: [fa_tc_v2b.cu](../../../mha_kernels/fa_tc_v2b.cu)

