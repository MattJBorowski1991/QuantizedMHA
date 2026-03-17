I. SRAM optimization (for Br=64)

I.1. Sram setup before any optimization (atomized sram buffers)

   __shared__ __align__(16) int8_t q_block[Br * (d + PAD)];
    __shared__ __align__(16) int8_t kt[d * (Bc + PAD)];

    __shared__ float scores_fp32[Br * (Bc + PAD)];
    __shared__ int scores_int32[Br * (Bc + PAD)];
    __shared__ __align__(16) int8_t scores_int8[Br * (Bc + PAD)];

    __shared__ int temp_output_int32[Br * (Bc + PAD)];
    __shared__ float output[Br * (d + PAD)];

    __shared__ __align__(16) int8_t values[Bc * (d + PAD)];

    __shared__ float sum_exp[Br];
    __shared__ float max_prev[Br];
    __shared__ float max_curr[Br];

For PAD=0 the statis SRAM allocation is:


Buffer	Dimensions	Data Type	Size
q_block	64 × 32	int8_t	2,048 B
kt	32 × 32	int8_t	1,024 B
scores_fp32	64 × 32	float	8,192 B
scores_int32	64 × 32	int	8,192 B
scores_int8	64 × 32	int8_t	2,048 B
temp_output_int32	64 × 32	int	8,192 B
output	64 × 32	float	8,192 B
values	32 × 32	int8_t	1,024 B
sum_exp	64	float	256 B
max_prev	64	float	256 B
max_curr	64	float	256 B
Total			39,680 B


I.2. Optimization attempts

I.2.1. union buffer for both scores_fp32 and scores_int32 => accuracy deteriorated by 0.3% (1-> 0.99604): 
Type aliasing issues: When the same memory location holds int32 data, then later holds float data, there can be compiler/hardware strict aliasing violations
Alignment and layout concerns: The int32 bit pattern doesn't map correctly to float interpretation, and overwrites can leave the buffer in an ambiguous state
CUDA's type safety: Passing the same buffer pointer as both int* and float* parameters can confuse the compiler's assumption about what's safe to optimize

I.2.2. same for applying a union buffer for scores_int32 and temp_output_int32 (1-> got=0.997797

I.2.3. Same for kt and values , however accuracy decrease is smaller :Mismatch at index: 7643392: got=0.997797 ref=1 tol=0.001

I.2.4. combining q_block with scores_int8 - did not cause any accuracy error! Difference? WMMA only touches one of them (q_block). WMMA operations have special hardware state/cache assumptions about their output buffers??

I.2.5. Force-checked pairing up all the largest buffers (8192b each) into a union buffer - none of the pairs worked due to either accuraccy or correcntess errors


II. SRAM allocation after introducing the two (small) union buffers: 

Buffer	Dimensions	Data Type	Size	Union	WMMA
q_block	64 × 32	int8_t	2,048 B	✅ 1	input
scores_int8	64 × 32	int8_t	2,048 B	✅ 1	output
kt	32 × 32	int8_t	1,024 B	✅ kv	input
values	32 × 32	int8_t	1,024 B	✅ kv	input
scores_fp32	64 × 32	float	8,192 B	❌	none
scores_int32	64 × 32	int	8,192 B	❌	output
temp_output_int32	64 × 32	int	8,192 B	❌	output
output	64 × 32	float	8,192 B	❌	none
sum_exp	64	float	256 B	❌	none
max_prev	64	float	256 B	❌	none
max_curr	64	float	256 B	❌	none
Total			36,608 B		


III. Subsequently we lower Br from 64 to 32, as noted in run6 (link) and change the tile from 16x16x16 to 8x16x32:

There is no accurac problem for 16x16x16 but there is a 0.4% error for 8x16x32!!!! And this is for both Br=64 and Br=32





. Extra notes

1. For occupancy what matters is Warps per SM and not Warps per Block!

2. For avoidance of doubt: 

 Padding is needed only in SRAM buffers -the output destinations for WMMA operations (q_block, scores_int8, kt, values, etc.) need padding for alignment and bank conflict avoidance.

DRAM inputs (Q, K, V) don't need padding—they're raw [N, d] layouts. Padding should only be added when writing to SRAM.


3. Debugging races when aliasing SRAM buffers.

3.1. Find shared memory hazards (test only for small N e.g. N=64)

compute-sanitizer --tool racecheck --racecheck-report all --racecheck-detect-level info --show-backtrace yes --force-blocking-launches --kernel-name kns=fa_kernel ./bin/profile_fa_tc_int8_b --warmup=0 --runs=1

3.2. Find Bad Barrier usage

compute-sanitizer --tool synccheck --show-backtrace yes --force-blocking-launches ./bin/profile_fa_tc_int8_b --warmup=0 --runs=1

3.3. Find reads of uninitialized memory

compute-sanitizer --tool initcheck --show-backtrace yes --force-blocking-launches ./bin/profile_fa_tc_int8_b --warmup=0 --runs=1




4. After aggressive SRAM optimizations - registers were the bottleneck for 5 block count. Optimizations to reduce register pressure: 
4.1. move max_new, sum_new, exp_max_diff into SRAM within online_softmax (before these were contributing 3x8 = 24 floats per thread)
4.2. remove pragma unrolls
4.2. remove __forceinline__ (or replace with noinline for large device kernels). forceinline = remove call overhead, increase register presure [use for small helpers]. noinline = add call overhead, reduce register pressure [use only for large helpers]. Blank/do nothing = uses compiler heuristics, which might inline anyways. -->> no improvement in duration
4.3. use __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor) = __launch_bounds__(THREADS, 2) which guides the compiler's register allocation strategy = I will never launch more than THREADS threads per block & I will have at least 2 blocks per SM. -->> improvement by ~1% in perf
4.4.  -maxregcount -->> didnt change regs per thread after launch_bounds found the optimal number of regs per thread (40), which verified with --ptxas-options=-v that it did not cause spilling: 

nvcc -O3 -lineinfo -Xcompiler -Wall --ptxas-options=-v -gencode arch=compute_89,code=sm_89 -gencode arch=compute_89,code=compute_89 -I. -I./include  drivers/main.cu inputs/data.cu utils/verify.cu mha_kernels/fa_tc_int8_b.cu -o bin/profile_fa_tc_int8_b
ptxas info    : 0 bytes gmem
ptxas info    : 0 bytes gmem
ptxas info    : 0 bytes gmem
ptxas info    : 267 bytes gmem, 40 bytes cmem[4]
ptxas info    : Compiling entry function '_Z9fa_kernelILi32ELi32ELi32ELi1EEvPKfS1_S1_PfS2_iif' for 'sm_89'
ptxas info    : Function properties for _Z9fa_kernelILi32ELi32ELi32ELi1EEvPKfS1_S1_PfS2_iif
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 40 registers, used 1 barriers, 6864 bytes smem, 404 bytes cmem[0], 16 bytes cmem[2]
ptxas info    : Compiling entry function '_Z10concat_matILi16EEvPfPKfiiiiii' for 'sm_89'
ptxas info    : Function properties for _Z10concat_matILi16EEvPfPKfiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 392 bytes cmem[0]
ptxas info    : Compiling entry function '_Z11extract_matILi16EEvPKfPfiiiiii' for 'sm_89'
ptxas info    : Function properties for _Z11extract_matILi16EEvPKfPfiiiiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 8 registers, used 0 barriers, 392 bytes cmem[0]
Built: bin/profile_fa_tc_int8_b (kernel=fa_tc_int8_b, arch=sm_89)

