// GEMM the Hard Way challenge edition.
// Day 26 blank focus: understand the final scheduling choice.
//
// fast.cu assigns output tiles to persistent CTAs using a Hilbert curve. The goal
// is to improve locality/load balance versus simple row-major tile assignment,
// especially for large square GEMMs where the same A/B regions are reused.
//
// FILL-IN BLANKS FOR THIS FILE
//   [ ] Blank A: trace d2xy(): how does a Hilbert distance become (m_tile,n_tile)?
//   [ ] Blank B: explain why createHilbert distributes early tiles over 64 cores.
//   [ ] Blank C: explain why the schedule is copied to device memory.
//   [ ] Blank D: benchmark 4096/8192 and compare to the fast.cu README numbers.
//   [ ] Blank E: propose one next change to close the remaining TFLOPS gap.

void sgemm_fastcu_hilbert_final_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    sgemm_fastcu_handwritten_cached_tma_maps_bf16(
        matrix_a, matrix_b_transposed, output_matrix_transposed);
}
