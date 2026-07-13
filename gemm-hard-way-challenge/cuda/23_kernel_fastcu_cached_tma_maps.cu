// GEMM the Hard Way challenge edition.
// Day 23 blank focus: tensor-map cache lifetime.
//
// Day 22 forced tensor-map recreation so a fresh output tensor was always
// correct. fast.cu's benchmark reuses stable A/B/C allocations, so tensor maps
// can be cached by pointer/shape and reused across timed iterations.
//
// Fill the solution in:
//   solutions/cuda/23_kernel_fastcu_cached_tma_maps.cu

void sgemm_fastcu_handwritten_cached_tma_maps_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    // BLANK A: extract M, N, K from A and B^T.
    const int M = GEMM_TODO_INT("Day23: M = A rows");
    const int K = GEMM_TODO_INT("Day23: K = A cols / B^T cols");
    const int N = GEMM_TODO_INT("Day23: N = B^T rows");

    // BLANK B: validate output C^T shape N x M.
    TORCH_CHECK(output_matrix_transposed.dim() == GEMM_TODO_INT("Day23: C^T rank"),
                "Output C^T must be 2D");
    (void)M;
    (void)K;
    (void)N;

    // BLANK C: unlike Day 22, do not reset M12::_prev_m before runKernel12().
    // The cached tensor maps are only safe when the wrapper reuses the same
    // output allocation during timing.
    output_matrix_transposed.zero_();
}
