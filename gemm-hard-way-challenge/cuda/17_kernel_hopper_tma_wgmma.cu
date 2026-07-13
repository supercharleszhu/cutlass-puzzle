// GEMM the Hard Way challenge edition.
// Day 17 blank focus: start the fast.cu H100 path with Hopper TMA + WGMMA.
// Replace/experiment with the marked constants after reading the fast.cu source.
#include "challenge_todo.cuh"


// FILL-IN BLANKS FOR THIS FILE
// fast.cu first moves from scalar/shared-memory math to Hopper primitives:
//
//   [ ] Blank A: map a CTA tile to the WGMMA tile family.
//   [ ] Blank B: explain why TMA needs 16-byte alignment and tensor-map strides.
//   [ ] Blank C: compare this CUTLASS TMA/WGMMA kernel with Day 15's schedules.
//   [ ] Blank D: profile tensor-core utilization before adding persistence.
//   [ ] Blank E: write down what remains memory/scheduler-bound.

#include <torch/torch.h>
#include <cuda_runtime.h>
#include "gemm_kernels.cuh"

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "cute/tensor.hpp"

using namespace cute;

namespace day17_standalone {

template <typename CollectiveMainloop, typename CollectiveEpilogue>
using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

struct Day17FastCuTmaWgmmaConfig
{
    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementC = cutlass::bfloat16_t;
    using ElementD = cutlass::bfloat16_t;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;
    using LayoutD = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

    using TileShape = Shape<_64, _64, _64>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized;
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator,
        ElementAccumulator,
        ElementC, LayoutC, AlignmentC,
        ElementD, LayoutD, AlignmentD,
        EpilogueSchedule>::CollectiveOp;

    using StageCountType = cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA, AlignmentA,
        ElementB, LayoutB, AlignmentB,
        ElementAccumulator,
        TileShape,
        ClusterShape,
        StageCountType,
        KernelSchedule>::CollectiveOp;

    using GemmKernel = day17_standalone::GemmKernel<CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

cudaError_t launch(
    int M, int N, int K,
    const cutlass::bfloat16_t *d_A,
    const cutlass::bfloat16_t *d_B,
    cutlass::bfloat16_t *d_D,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    using Config = Day17FastCuTmaWgmmaConfig;
    typename Config::Gemm gemm_op;
    auto problem_shape = make_shape(M, N, K);

    using StrideA = typename Config::GemmKernel::StrideA;
    using StrideB = typename Config::GemmKernel::StrideB;
    using StrideC = typename Config::GemmKernel::StrideC;
    using StrideD = typename Config::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename Config::Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {d_A, stride_A, d_B, stride_B},
        {{1.0f, 0.0f}, d_D, stride_C, d_D, stride_D},
        hw_info
    };

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;

    size_t workspace_size = Config::Gemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t result = cudaMalloc(&workspace, workspace_size);
        if (result != cudaSuccess)
            return result;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess)
        status = gemm_op.run(stream);

    if (workspace)
        cudaFree(workspace);

    return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

void torch_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    TORCH_CHECK(matrix_a.device().is_cuda(), "Matrix A must be on CUDA device");
    TORCH_CHECK(matrix_b.device().is_cuda(), "Matrix B must be on CUDA device");
    TORCH_CHECK(output_matrix.device().is_cuda(), "Output matrix must be on CUDA device");

    TORCH_CHECK(matrix_a.scalar_type() == at::kBFloat16, "Matrix A must be bfloat16");
    TORCH_CHECK(matrix_b.scalar_type() == at::kBFloat16, "Matrix B must be bfloat16");
    TORCH_CHECK(output_matrix.scalar_type() == at::kBFloat16, "Output matrix must be bfloat16");

    TORCH_CHECK(matrix_a.dim() == 2 && matrix_b.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(matrix_a.is_contiguous() && matrix_b.is_contiguous(),
                "Input tensors must be contiguous for TMA alignment requirements");
    TORCH_CHECK(output_matrix.is_contiguous(), "Output tensor must be contiguous");

    const int M = static_cast<int>(matrix_a.size(0));
    const int K = static_cast<int>(matrix_a.size(1));
    const int N = static_cast<int>(matrix_b.size(1));

    TORCH_CHECK(matrix_b.size(0) == K, "Matrix dimension mismatch");
    TORCH_CHECK(output_matrix.size(0) == M && output_matrix.size(1) == N,
                "Output matrix has wrong shape");

    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_a.data_ptr()) % 16 == 0,
                "Matrix A must be 16-byte aligned for Hopper TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(matrix_b.data_ptr()) % 16 == 0,
                "Matrix B must be 16-byte aligned for Hopper TMA");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(output_matrix.data_ptr()) % 16 == 0,
                "Output matrix must be 16-byte aligned for Hopper TMA");

    const auto *d_A = reinterpret_cast<const cutlass::bfloat16_t *>(matrix_a.data_ptr<at::BFloat16>());
    const auto *d_B = reinterpret_cast<const cutlass::bfloat16_t *>(matrix_b.data_ptr<at::BFloat16>());
    auto *d_D = reinterpret_cast<cutlass::bfloat16_t *>(output_matrix.data_ptr<at::BFloat16>());

    const cudaError_t err = launch(M, N, K, d_A, d_B, d_D);
    TORCH_CHECK(err == cudaSuccess, "Day 17 Hopper TMA/WGMMA GEMM failed: ", cudaGetErrorString(err));
}

} // namespace day17_standalone

void sgemm_cutlass_hopper_fastcu_tma_wgmma_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    day17_standalone::torch_wrapper(matrix_a, matrix_b, output_matrix);
}
