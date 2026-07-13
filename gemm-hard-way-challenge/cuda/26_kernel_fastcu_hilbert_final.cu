// GEMM the Hard Way challenge edition.
// Day 26 blank focus: Hilbert scheduling and final handwritten kernel analysis.
//
// The solution uses createHilbert() to generate a spatial tile order for
// persistent CTAs. Consecutive tiles stay close in M/N space, improving L2 reuse.

void sgemm_fastcu_hilbert_final_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    // BLANK A: trace d2xy() in the solution: distance -> (m_tile, n_tile).
    // BLANK B: explain why the schedule is copied to device memory.
    // BLANK C: call the cached-map Day 23 kernel after understanding the
    // schedule path.
    (void)matrix_a;
    (void)matrix_b_transposed;
    output_matrix_transposed.zero_();
}
