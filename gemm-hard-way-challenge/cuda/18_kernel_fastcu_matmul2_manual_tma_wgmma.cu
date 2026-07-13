// GEMM the Hard Way challenge edition.
// Day 18 blank focus: handwritten fast.cu matmul_2-style Hopper TMA + WGMMA.
//
// Read the reference:
//   https://github.com/pranjalssh/fast.cu/blob/main/examples/matmul/matmul_2.cuh
//
// This day sits between the CUTLASS Day 17 TMA/WGMMA baseline and the larger
// Day 19 tile. The goal is to manually identify the same Hopper pieces that
// CUTLASS hid on Day 17:
//   1. encode WGMMA shared-memory descriptors,
//   2. create 2D TMA tensor maps for A and B^T,
//   3. issue cp.async.bulk.tensor global-to-shared copies,
//   4. run wgmma.m64n64k16 over four BK=16 slices,
//   5. store the 64x64 accumulator tile into C^T.
#include "challenge_todo.cuh"


// FILL-IN BLANKS FOR THIS FILE
//   [ ] Blank A: explain why matmul_2 uses one 128-thread warpgroup per CTA.
//   [ ] Blank B: trace the TMA tensor map shapes for row-major A and B^T.
//   [ ] Blank C: explain the barrier arrive/wait sequence around TMA loads.
//   [ ] Blank D: map the four m64n64k16 WGMMA calls to BK=64.
//   [ ] Blank E: explain why this early handwritten kernel writes C^T with
//       per-thread stores instead of the later TMA-store epilogue.

void sgemm_fastcu_matmul2_manual_tma_wgmma_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    (void)matrix_a;
    (void)matrix_b_transposed;
    output_matrix_transposed.zero_();
}
