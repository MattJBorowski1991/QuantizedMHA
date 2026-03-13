I. SRAM optimization

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

I.2.4. combining q_block with scores_int8 - did not cause any accuracy error!

