# GEMM the Hard Way Standalone Challenge

This folder is a standalone exercise version of the first blog's CUDA path, adapted from the Apache-2.0 `gpusgobrr/explore-gemm` implementation:

- Blog: https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/
- Source: https://github.com/gpusgobrr/explore-gemm/tree/main/cuda

Each CUDA file from `01` through `14` is included under `cuda/`. The code keeps the surrounding implementation but replaces selected key expressions with `GEMM_TODO_*` placeholders so you can fill in the optimization step yourself.

## Layout

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
| `cuda/challenge_todo.cuh` | Placeholder macros used by the exercise blanks. |
| `BLANKS.md` | Greppable TODO list grouped by file. |
| `solutions/cuda/` | Complete 01-14 solution kernels copied from the upstream implementation. |
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

## Solutions

The `solutions/` folder contains the complete 01-14 implementation and a minimal runner:

```bash
cd gemm-hard-way-challenge
python3 solutions/python/benchmark_solution.py --sizes 128 256 --dtype float32
python3 solutions/python/benchmark_solution.py --sizes 128 256 --dtype float16
```

By default the solution runner builds files 01-12. To also build CUTLASS files 13-14, run from a LeetCUDA checkout that has `./cutlass` populated, or set `CUTLASS_DIR`, then pass `--include-cutlass`.

For remote GPU execution through Kubernetes, see [`REMOTE_POD_GUIDE.md`](./REMOTE_POD_GUIDE.md).

Quick remote examples:

```bash
cd gemm-hard-way-challenge
scripts/run_remote_matrix.sh --day 1
scripts/run_remote_matrix.sh --day 9 --sizes "128 256" --iters 50
scripts/run_remote_matrix.sh --day 14 --with-cutlass
scripts/run_remote_matrix.sh --all --skip-upload
```

## Attribution

The upstream implementation is licensed under Apache License 2.0; a copy is included as `LICENSE.upstream`.
