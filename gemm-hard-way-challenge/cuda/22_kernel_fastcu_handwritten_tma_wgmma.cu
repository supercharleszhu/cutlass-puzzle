// GEMM the Hard Way challenge edition.
// Day 22 blank focus: handwritten WGMMA + TMA skeleton.
//
// Read the full solution in:
//   solutions/cuda/22_kernel_fastcu_handwritten_tma_wgmma.cu
//
// This first handwritten day is about identifying the essential pieces of the
// fast.cu final kernel before worrying about benchmarking:
//   1. shared-memory descriptor encoding for WGMMA,
//   2. m64n256k16 BF16->FP32 WGMMA,
//   3. TMA tensor-map creation,
//   4. the B^T / C^T layout bridge used by the wrapper.

using std::max;
typedef __nv_bfloat16 bf16;

namespace fastcu_day21 {

// BLANK A: shared-memory descriptors feed WGMMA. Fill in the same descriptor
// fields as the solution: base address, leading dimension, stride, and 128B
// swizzle bit.
__device__ static inline uint64_t make_smem_desc(bf16* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = GEMM_TODO_UINT("Day22: encode shared-memory base pointer");
    desc |= static_cast<uint64_t>(GEMM_TODO_UINT("Day22: encode leading byte offset")) << 16;
    desc |= static_cast<uint64_t>(GEMM_TODO_UINT("Day22: encode stride byte offset")) << 32;
    desc |= static_cast<uint64_t>(GEMM_TODO_UINT("Day22: set 128B swizzle bit")) << 62;
    (void)addr;
    return desc;
}

// BLANK B: the real solution wraps wgmma.mma_async.sync.aligned.m64n256k16.
// Fill the PTX operand list in the solution file after understanding how the
// 128-thread warpgroup owns a 64x256 accumulator tile.
__device__ __forceinline__ void wgmma256(float d[16][8], bf16* sA, bf16* sB) {
    (void)d;
    (void)sA;
    (void)sB;
    GEMM_TODO_WMMA_MMA("Day22: emit WGMMA m64n256k16 BF16->FP32 instruction");
}

// BLANK C: create the 3D TMA map used by the final kernel. The solution uses
// a first tensor dimension of 64 columns, height as rows, and width/64 as the
// third dimension so TMA can move 128x256-style tiles.
template <int BlockMajorSize, int BlockMinorSize, bool swizzle=true, bool padding=false>
__host__ static inline CUtensorMap create_tensor_map(bf16* gmem_ptr, int global_height, int global_width) {
    CUtensorMap tma_map{};
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[5] = {
        static_cast<uint64_t>(GEMM_TODO_INT("Day22: TMA shape dim0 columns-per-slice")),
        static_cast<uint64_t>(global_height),
        static_cast<uint64_t>(GEMM_TODO_INT("Day22: TMA shape dim2 width/64")),
        1, 1};
    uint64_t gmem_prob_stride[5] = {
        sizeof(bf16) * static_cast<uint64_t>(global_width),
        static_cast<uint64_t>(GEMM_TODO_INT("Day22: TMA stride between 64-column slices")),
        0, 0, 0};
    uint32_t smem_box_shape[5] = {
        static_cast<uint32_t>(GEMM_TODO_INT("Day22: SMEM box first dim, 64 or padded 72")),
        uint32_t(BlockMajorSize),
        uint32_t(BlockMinorSize / 64),
        1, 1};
    uint32_t smem_box_stride[5] = {1, 1, 1, 1, 1};
    (void)gmem_address;
    (void)gmem_prob_shape;
    (void)gmem_prob_stride;
    (void)smem_box_shape;
    (void)smem_box_stride;
    (void)swizzle;
    (void)padding;
    return tma_map;
}

} // namespace fastcu_day21

// BLANK D: bridge the row-major PyTorch benchmark to fast.cu's B^T/C^T kernel
// convention. The solution validates A, B^T, and C^T, then calls runKernel12().
void sgemm_fastcu_handwritten_tma_wgmma_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    (void)matrix_a;
    (void)matrix_b_transposed;
    output_matrix_transposed.zero_();
}
