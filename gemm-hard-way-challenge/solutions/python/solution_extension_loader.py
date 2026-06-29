"""Build the complete GEMM hard-way solution kernels as a PyTorch extension."""

from __future__ import annotations

import os
from pathlib import Path

from torch.utils.cpp_extension import load_inline


CUDA_FILES_BASE = [
    "01_naive.cu",
    "02_kernel_global_mem_coalesce.cu",
    "03_kernel_shared_mem.cu",
    "04_kernel_blocktiling_1d.cu",
    "05_kernel_blocktiling_2d.cu",
    "06_kernel_vectorize.cu",
    "07_kernel_warptiling.cu",
    "08_kernel_warptiling_all_dtypes.cu",
    "09_kernel_tensorcore_naive.cu",
    "10_kernel_tensorcore_warptiled.cu",
    "11_kernel_tensorcore_double_buffered.cu",
    "12_kernel_tensorcore_async.cu",
]

CUDA_FILES_CUTLASS = [
    "13_kernel_cutlass.cu",
    "14_kernel_cutlass_autotunable.cu",
]

FUNCTIONS_BASE = [
    "sgemm_naive",
    "sgemm_global_mem_coalesce",
    "sgemm_shared_mem",
    "sgemm_blocktiling_1d",
    "sgemm_blocktiling_2d",
    "sgemm_vectorize",
    "sgemm_warptiling_default",
    "sgemm_warptiling_fp16",
    "sgemm_warptiling_bf16",
    "sgemm_tensorcore_naive_fp16",
    "sgemm_tensorcore_naive_bf16",
    "sgemm_tensorcore_fp16",
    "sgemm_tensorcore_bf16",
    "sgemm_tensorcore_double_buffered_fp16",
    "sgemm_tensorcore_double_buffered_bf16",
    "sgemm_tensorcore_async_fp16",
    "sgemm_tensorcore_async_bf16",
]

FUNCTIONS_CUTLASS = [
    "sgemm_cutlass_fp16",
    "sgemm_cutlass_bf16",
    "sgemm_cutlass_fp32",
    "sgemm_cutlass_autotune_fp16",
    "sgemm_cutlass_autotune_bf16",
    "get_num_cutlass_configs",
]


def _read_without_local_includes(path: Path) -> str:
    lines = []
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith("#include"):
            continue
        if line.startswith("#pragma once"):
            continue
        lines.append(line)
    return "".join(lines)


def _find_cutlass_paths() -> list[str]:
    candidates = []
    if os.environ.get("CUTLASS_DIR"):
        candidates.append(Path(os.environ["CUTLASS_DIR"]))
    if os.environ.get("CUTLASS_INCLUDE_DIR"):
        candidates.append(Path(os.environ["CUTLASS_INCLUDE_DIR"]).parent)

    here = Path(__file__).resolve()
    # A local ./cutlass checkout if this repository is used standalone.
    candidates.append(here.parents[2] / "cutlass")
    # A sibling LeetCUDA checkout, useful for local development.
    candidates.append(here.parents[2].parent / "LeetCUDA" / "cutlass")
    # Optional colocated copy/symlink for a standalone upload.
    candidates.append(here.parents[1] / "third-party" / "cutlass")

    for base in candidates:
        include = base / "include"
        util_include = base / "tools" / "util" / "include"
        if (include / "cutlass").exists():
            paths = [str(include)]
            if util_include.exists():
                paths.append(str(util_include))
            return paths
    return []


def create_solution_extension(
    *,
    include_cutlass: bool = False,
    verbose: bool = False,
):
    """Compile and load the solution extension.

    By default this builds files 01-12, which do not require CUTLASS headers.
    Pass include_cutlass=True to also build files 13-14.
    """

    file_dir = Path(__file__).resolve().parents[1] / "cuda"
    header_code = _read_without_local_includes(file_dir / "gemm_kernels.cuh")
    utils_code = _read_without_local_includes(file_dir / "utils.cuh")

    files = list(CUDA_FILES_BASE)
    functions = list(FUNCTIONS_BASE)
    extra_include_paths: list[str] = []

    if include_cutlass:
        extra_include_paths = _find_cutlass_paths()
        if not extra_include_paths:
            raise RuntimeError(
                "CUTLASS headers were not found. Set CUTLASS_DIR or run from a "
                "LeetCUDA checkout that has ./cutlass populated."
            )
        files += CUDA_FILES_CUTLASS
        functions += FUNCTIONS_CUTLASS

    cuda_header = r"""
#ifdef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_OPERATORS__
#endif
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#ifdef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#endif
#ifdef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF2_OPERATORS__
#endif

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <iostream>
#include <type_traits>
namespace cg = cooperative_groups;
"""

    if include_cutlass:
        cuda_header += r"""
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
"""

    cuda_sources = cuda_header + "\n" + utils_code + "\n"
    for name in files:
        cuda_sources += "\n" + _read_without_local_includes(file_dir / name)

    build_dir = file_dir.parent / "build" / ("solution_cutlass" if include_cutlass else "solution_base")
    build_dir.mkdir(parents=True, exist_ok=True)

    return load_inline(
        name="gemm_hard_way_solution_cutlass" if include_cutlass else "gemm_hard_way_solution",
        cpp_sources=header_code,
        cuda_sources=cuda_sources,
        functions=functions,
        with_cuda=True,
        verbose=verbose,
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        extra_include_paths=extra_include_paths,
        build_directory=str(build_dir),
    )
