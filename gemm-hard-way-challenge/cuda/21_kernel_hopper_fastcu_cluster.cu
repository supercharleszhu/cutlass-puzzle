// GEMM the Hard Way challenge edition.
// Day 21 blank focus: add the fast.cu-style 2x1 CTA cluster on top of persistence.
#include "challenge_todo.cuh"


// FILL-IN BLANKS FOR THIS FILE
// fast.cu's later kernels use clusters and TMA multicast so neighboring CTAs can
// share one operand tile. This CUTLASS version exposes the same design point.
//
//   [ ] Blank A: explain why ClusterShape<2,1,1> shares A across two M tiles.
//   [ ] Blank B: compare 1x1 and 2x1 clusters for large square BF16 GEMMs.
//   [ ] Blank C: profile TMA traffic and tensor-core utilization.
//   [ ] Blank D: compare against Day 16 Stream-K and Day 20 persistent-only.
//   [ ] Blank E: read fast.cu matmul_12 and note what handwritten TMA store and
//       Hilbert scheduling add beyond this CUTLASS version.

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

namespace day21_standalone {

enum class SchedulerKind {
    OneCtaPerTile,
    Persistent,
};

template <typename Scheduler, typename CollectiveMainloop, typename CollectiveEpilogue>
static auto make_gemm_kernel_type()
{
    if constexpr (std::is_void_v<Scheduler>)
    {
        return cutlass::gemm::kernel::GemmUniversal<
            cute::Shape<int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue>{};
    }
    else
    {
        return cutlass::gemm::kernel::GemmUniversal<
            cute::Shape<int, int, int>,
            CollectiveMainloop,
            CollectiveEpilogue,
            Scheduler>{};
    }
}

template <
    int TileM,
    int TileN,
    int TileK,
    int ClusterM,
    int ClusterN,
    SchedulerKind SchedulerMode>
struct HopperFastCuConfig
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

    using TileShape = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
    using ClusterShape = cute::Shape<cute::Int<ClusterM>, cute::Int<ClusterN>, cute::_1>;
    using KernelSchedule = std::conditional_t<
        SchedulerMode == SchedulerKind::Persistent || (ClusterM * ClusterN > 1),
        cutlass::gemm::KernelTmaWarpSpecializedCooperative,
        cutlass::gemm::KernelTmaWarpSpecialized>;
    using EpilogueSchedule = std::conditional_t<
        SchedulerMode == SchedulerKind::Persistent || (ClusterM * ClusterN > 1),
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        cutlass::epilogue::TmaWarpSpecialized>;
    using TileSchedulerType = std::conditional_t<
        SchedulerMode == SchedulerKind::Persistent,
        cutlass::gemm::PersistentScheduler,
        void>;

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

    using GemmKernel = decltype(make_gemm_kernel_type<TileSchedulerType, CollectiveMainloop, CollectiveEpilogue>());
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename Config>
cudaError_t launch(
    int M, int N, int K,
    const typename Config::ElementA *d_A,
    const typename Config::ElementB *d_B,
    typename Config::ElementD *d_D,
    cudaStream_t stream = nullptr)
{
    if (M == 0 || N == 0 || K == 0)
        return cudaSuccess;

    typename Config::Gemm gemm_op;
    auto problem_shape = cute::make_shape(M, N, K);

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

    float alpha = 1.0f;
    float beta = 0.0f;

    typename Config::Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {d_A, stride_A, d_B, stride_B},
        {{alpha, beta}, d_D, stride_C, d_D, stride_D},
        hw_info
    };

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        return cudaErrorNotSupported;

    size_t workspace_size = Config::Gemm::get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0)
    {
        cudaError_t result = cudaMalloc(&workspace, workspace_size);
        if (result != cudaSuccess)
            return result;
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status == cutlass::Status::kSuccess)
        status = gemm_op.run(stream);

    if (workspace)
        cudaFree(workspace);

    if (status != cutlass::Status::kSuccess)
        return cudaErrorUnknown;
    return cudaSuccess;
}

template <typename Config>
void torch_wrapper(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix,
    const char *kernel_name)
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

    const cudaError_t err = launch<Config>(M, N, K, d_A, d_B, d_D);
    TORCH_CHECK(err == cudaSuccess, kernel_name, " failed: ", cudaGetErrorString(err));
}

} // namespace day21_standalone


using Day21FastCuClusterConfig = day21_standalone::HopperFastCuConfig<
    128, 256, 64,
    2, 1,
    day21_standalone::SchedulerKind::Persistent>;

void sgemm_cutlass_hopper_fastcu_cluster_bf16(
    const torch::Tensor &matrix_a,
    const torch::Tensor &matrix_b,
    torch::Tensor &output_matrix)
{
    day21_standalone::torch_wrapper<Day21FastCuClusterConfig>(
        matrix_a, matrix_b, output_matrix, "Day 21 Hopper 2x1 cluster GEMM");
}
