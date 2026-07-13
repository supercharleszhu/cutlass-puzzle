// GEMM the Hard Way challenge edition.
// Day 23 blank focus: match fast.cu's benchmarking assumption that A/B/C
// allocations stay stable, so TMA tensor maps can be cached across timed runs.
//
// FILL-IN BLANKS FOR THIS FILE
//   [ ] Blank A: explain why Day 22 forced tensor-map recreation for correctness.
//   [ ] Blank B: explain why recreating tensor maps inside timed iterations is
//       not comparable to fast.cu's benchmark.
//   [ ] Blank C: trace how the Python wrapper caches B^T and C^T by input pointer.
//   [ ] Blank D: benchmark Day 22 vs Day 23 at N=4096.
//   [ ] Blank E: note why this wrapper is benchmark-oriented, not a general API.

void sgemm_fastcu_handwritten_cached_tma_maps_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b_transposed,
    torch::Tensor &output_matrix_transposed)
{
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b_transposed.device().is_cuda(), "Matrix B^T must be on CUDA device");
    TORCH_CHECK(output_matrix_transposed.device().is_cuda(), "Output C^T must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == at::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b_transposed.scalar_type() == at::kBFloat16, "Matrix B^T must be bfloat16");
    TORCH_CHECK(output_matrix_transposed.scalar_type() == at::kBFloat16, "Output C^T must be bfloat16");

    const int M = static_cast<int>(matrix_a.size(0));
    const int K = static_cast<int>(matrix_a.size(1));
    const int N = static_cast<int>(matrix_b_transposed.size(0));
    TORCH_CHECK(matrix_b_transposed.size(1) == K, "B^T must have shape N x K");
    TORCH_CHECK(output_matrix_transposed.size(0) == N && output_matrix_transposed.size(1) == M,
                "Output C^T must have shape N x M");

    auto *d_A = reinterpret_cast<bf16 *>(matrix_a.data_ptr<at::BFloat16>());
    auto *d_Bt = reinterpret_cast<bf16 *>(matrix_b_transposed.data_ptr<at::BFloat16>());
    auto *d_Ct = reinterpret_cast<bf16 *>(output_matrix_transposed.data_ptr<at::BFloat16>());

    runKernel12(M, N, K, d_A, d_Bt, d_Ct, nullptr);
}
