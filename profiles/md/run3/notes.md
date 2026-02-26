


- You cannot / should not assign work to lanes in Tensor Cores, as warp is the atomic execution unit

- You can assign work to lanes for double-buffering

- first TC implementation: one warp owns (Wr x d) = (16 x d) of Q and performs a serial mma of 16x16 tiles

- e.g. for Q x Kt the only non-serial application will be the multiplications of the first warp-owned-tiles of Q of 16x16 each with the first warp-owned-tiles of Kt of 16x16 each as the rest is serial. This is equivalent to doing non-serial wmma only on (Br x 16) of Q and (Bc x 16) of K.

- We are limited on L4 by the SRAM to Br = 64 so only 64/16 = 4 warps per block

- In normal TC WMMA for GEMM one block can even be responsible only for a 16x16 tile (1 block = 1 warp = 32 threads) and hence all the mamtul is done in waves execution. Here in Flash Attention however we are limited by the fact that Br cannot be too large due to SRAM (on L4 anything for Br > 64 is risky) and by the fact that one warp can only handle tile sizes of: 16x16x16, 16x16x8, 32x8x16, 8x32x16, 8x8x32 (MxNxK). For this reason as our first version of TC we take the standard 16x16(x16) tile size and do not split the d-dimension across many warps => one warp owns 16xd of Q in a serialized way. 

- In the next versions of the Tensor Core application we can should try 8x8x32 tile size to increase the number of warps per block.