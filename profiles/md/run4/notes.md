0. assumption is that we want more warps to "fit" in the Br-dimension, so we take the wmma tile size of 8x32x16 instead of 16x16x16

1. first approach to warp-work distribution accross the d-dimension: splitting the work between 2 warps = one owns the left half, the other owns the right half. This way with Br=64 and a 8x32x16 Tensor Core tile we have 16 warps working instead of 8 before. 

2. Need to accumulate the "left" result and the "right" result and at the end add them together.

3. First approach to accumulation: SRAM. Each of left/right buffers requires the following amountof SRAM: 

3.1. Q@K^T: 2 * Br * (d + PAD) = c.a. 32 kB for PAD = 32 (necessary due to 8x32x16 wmma tile size)
3.2. P @ V: 2 * Br * (Bc + PAD) = c.a. 32kB

This is A LOT and causes SRAM overflow and the kernel failing silently.

4. Bug for mismatch with cpu test reference: error was occuring at random output indices (sometimes on far output indices):
was caused by initial races within online_softmax_and_accum_output, as both "left" and "right" warps were racing to write into the same indices. adding "if (warp_tile_col_id == 0) " and small refactor removed the race. 

