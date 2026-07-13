# GEMM the Hard Way Standalone Challenge

This folder is a standalone exercise path for learning GEMM implementation in two stages:

1. Days 1-14 introduce the CUDA and CUTLASS concepts used by the original `gpusgobrr/explore-gemm` CUDA path.
2. Days 15-26 move to Hopper-specific kernels: CUTLASS 3.x TMA/WGMMA first, then handwritten fast.cu-derived challenge days.

References:

- CUTLASS intro blog: https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/
- explore-gemm source: https://github.com/gpusgobrr/explore-gemm/tree/main/cuda
- CUTLASS official SM90 kernel headers: https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/gemm/kernel
- Hopper fast.cu-style blog notes: BF16 H100 matmul path with WGMMA, TMA, persistent scheduling, clusters, and L2-aware scheduling.

Each CUDA file from `01` through `26` is included under `cuda/`. Days 1-14 keep selected `GEMM_TODO_*` expression blanks. Days 15-17 are CUTLASS/Hopper design puzzles, Day 18 introduces handwritten fast.cu matmul_2-style TMA/WGMMA, Days 19-21 continue the CUTLASS-facing fast.cu schedule analogues, and Days 22-26 walk through the later handwritten fast.cu path.

## Day map

### Days 1-14: CUDA-to-CUTLASS introduction

| Path | Purpose |
| --- | --- |
| `cuda/01_naive.cu` | Naive FP32 GEMM baseline. |
| `cuda/02_kernel_global_mem_coalesce.cu` | Coalesced thread-to-output mapping. |
| `cuda/03_kernel_shared_mem.cu` | Shared-memory A/B tiling. |
| `cuda/04_kernel_blocktiling_1d.cu` | 1D register tiling. |
| `cuda/05_kernel_blocktiling_2d.cu` | 2D per-thread register tile. |
| `cuda/06_kernel_vectorize.cu` | Vectorized `float4` loads and SMEM layout. |
| `cuda/07_kernel_warptiling.cu` | CTA/warp/thread tiling hierarchy. |
| `cuda/08_kernel_warptiling_all_dtypes.cu` | FP32/FP16/BF16 warp tiling. |
| `cuda/09_kernel_tensorcore_naive.cu` | Naive WMMA Tensor Core baseline. |
| `cuda/10_kernel_tensorcore_warptiled.cu` | WMMA plus block/warp tiling. |
| `cuda/11_kernel_tensorcore_double_buffered.cu` | Tensor Core double buffering. |
| `cuda/12_kernel_tensorcore_async.cu` | Async pipeline variant. |
| `cuda/13_kernel_cutlass.cu` | CUTLASS 2.x GEMM wrapper. |
| `cuda/14_kernel_cutlass_autotunable.cu` | CUTLASS autotuning configs. |

### Days 15-26: Hopper optimization with CUTLASS and handwritten fast.cu machinery

These days use CUTLASS 3.x APIs that instantiate `cutlass::gemm::kernel::GemmUniversal` from the official `include/cutlass/gemm/kernel` SM90 family:

- `sm90_gemm_tma_warpspecialized.hpp`
- `sm90_gemm_tma_warpspecialized_pingpong.hpp`
- `sm90_gemm_tma_warpspecialized_cooperative.hpp`
- `tile_scheduler.hpp` / `tile_scheduler_params.h`

The sequence mirrors the pasted H100 blog at the level CUTLASS exposes directly:

| Path | Hopper focus |
| --- | --- |
| `cuda/15_kernel_cutlass_hopper.cu` | Choose between CUTLASS SM90 TMA warp-specialized schedules: basic, persistent/cooperative, ping-pong, and Stream-K. |
| `cuda/16_kernel_cutlass_hopper_autotunable.cu` | Expose tile shape, raster order, decomposition mode, swizzle, and split count as runtime autotuning knobs. |
| `cuda/17_kernel_hopper_tma_wgmma.cu` | Blog kernel 2 analogue: first BF16 Hopper TMA + WGMMA tile through CUTLASS `GemmUniversal`. |
| `cuda/18_kernel_fastcu_matmul2_manual_tma_wgmma.cu` | fast.cu matmul_2 analogue: manually apply 2D TMA loads, barriers, WGMMA m64n64k16, and C^T stores. |
| `cuda/19_kernel_hopper_fastcu_big_tile.cu` | Blog kernel 3/5 analogue: scale to a 128x256x64 tile to increase WGMMA work per CTA. |
| `cuda/20_kernel_hopper_fastcu_persistent.cu` | Blog kernel 6 analogue: persistent scheduling so CTAs pull multiple output tiles and improve residency/load balance. |
| `cuda/21_kernel_hopper_fastcu_cluster.cu` | Blog kernel 8 analogue: 2x1 CTA cluster using cooperative SM90 schedules, the CUTLASS-facing version of cluster/TMA-multicast reasoning. |
| `cuda/22_kernel_fastcu_handwritten_tma_wgmma.cu` | Blog kernel 12 analogue: handwritten WGMMA/TMA, TMA store, stmatrix staging, and Hilbert scheduling. |
| `cuda/23_kernel_fastcu_cached_tma_maps.cu` | fast.cu benchmark assumption: stable A/B/C allocations and cached TMA tensor maps. |
| `cuda/24_kernel_fastcu_final.cu` | final fast.cu-style benchmark target for 4096/8192 comparisons. |
| `cuda/25_kernel_fastcu_tma_store.cu` | handwritten epilogue focus: stmatrix into shared memory followed by TMA store. |
| `cuda/26_kernel_fastcu_hilbert_final.cu` | handwritten scheduling focus: Hilbert tile order and final benchmark analysis. |
| `LICENSE.fastcu` | MIT license for the fast.cu-derived source. |

The handwritten fast.cu ports use fast.cu's B^T/C^T layout convention: the Python wrapper passes `B.t().contiguous()` to the kernel and returns a transposed view of the output buffer.

## Layout

| Path | Purpose |
| --- | --- |
| `cuda/challenge_todo.cuh` | Placeholder macros used by the exercise blanks. |
| `BLANKS.md` | Greppable TODO list grouped by file. |
| `solutions/cuda/` | Complete solution kernels copied/adapted from the upstream implementation and later Hopper exercises. |
| `solutions/python/benchmark_solution.py` | Correctness and benchmark runner for the complete solutions. |
| `scripts/upload_to_pod.sh` | Upload this standalone challenge to the configured Kubernetes pod. |
| `scripts/run_remote_day.sh` | Run correctness + benchmark for one or more days on the pod. |
| `scripts/run_remote_matrix.sh` | Convenience wrapper that uploads, then runs selected days. |
| `REMOTE_POD_GUIDE.md` | How to copy the challenge to a Kubernetes GPU pod and run correctness, benchmark, and profiling. |

## How to use

1. Open one file at a time, starting with `cuda/01_naive.cu`.
2. Search for `GEMM_TODO`.
3. Replace the placeholder with the real expression from the blog concept.
4. Run correctness against `torch.matmul`.
5. Benchmark and profile before moving to the next file.

The placeholder macros currently return dummy values so the blanks are easy to find. The exercise kernels are not expected to be correct until you replace the placeholders. Each exercise file also has a small fill-in checklist near the top so you can write down the mapping, load, compute, store, and profiling blanks for that stage.

For CUTLASS days, remember that `GemmShape<M, N, K>` still follows GEMM dimensions: `M` rows of C/A, `N` cols of C/B, and `K` reduction depth. In Day 13, `ThreadBlockShape = GemmShape<BM, BN, BK>` is the CTA tile, not the full problem size.

## Solutions

The `solutions/` folder contains complete implementations and a minimal runner:

```bash
cd gemm-hard-way-challenge
python3 solutions/python/benchmark_solution.py --sizes 128 256 --dtype float32
python3 solutions/python/benchmark_solution.py --sizes 128 256 --dtype float16
```

By default the solution runner builds files 01-12. To also build CUTLASS files 13-14, run from a LeetCUDA checkout that has `./cutlass` populated, or set `CUTLASS_DIR`, then pass `--include-cutlass`. On Hopper GPUs (SM90+), `--include-cutlass` also builds days 15-26 with `sm_90a`.

For remote GPU execution through Kubernetes, see [`REMOTE_POD_GUIDE.md`](./REMOTE_POD_GUIDE.md).

Quick remote examples:

```bash
cd gemm-hard-way-challenge
scripts/run_remote_matrix.sh --day 1
scripts/run_remote_matrix.sh --day 9 --sizes "128 256" --iters 50
scripts/run_remote_matrix.sh --day 14 --with-cutlass
scripts/upload_run_one_day.sh --day 15 --with-cutlass
scripts/upload_run_one_day.sh --day 18 --with-cutlass --sizes "1024 2048 4096"
scripts/upload_run_one_day.sh --day 21 --with-cutlass --sizes "1024 2048 4096"
scripts/upload_run_one_day.sh --day 22 --with-cutlass --sizes "4096 8192"
scripts/upload_run_one_day.sh --day 26 --with-cutlass --sizes "4096 8192"
scripts/run_remote_matrix.sh --all --skip-upload
```

## Attribution

The original explore-gemm implementation is licensed under Apache License 2.0; a copy is included as `LICENSE.upstream`.
Days 18 and 22-26 include code or exercises adapted from the MIT-licensed fast.cu repository; the license is included as `LICENSE.fastcu`.
