// GEMM the Hard Way challenge edition.
// Day 25 blank focus: understand fast.cu's handwritten epilogue.
//
// The final handwritten kernel does not store accumulator registers directly to
// global memory. It first uses stmatrix to put the WGMMA accumulator fragment
// into a shared-memory layout accepted by TMA store, then issues
// cp.async.bulk.tensor.*.global.shared.
//
// FILL-IN BLANKS FOR THIS FILE
//   [ ] Blank A: find where accumulator registers are converted to bf16.
//   [ ] Blank B: explain the stmatrix address calculation and the +8 row padding.
//   [ ] Blank C: explain why cp.async.bulk.wait_group is before the TMA store.
//   [ ] Blank D: compare TMA store with per-thread global stores.
//   [ ] Blank E: profile store-side stalls and global-store throughput.

void sgemm_fastcu_tma_store_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    sgemm_fastcu_handwritten_cached_tma_maps_bf16(
        matrix_a, matrix_b_transposed, output_matrix_transposed);
}
