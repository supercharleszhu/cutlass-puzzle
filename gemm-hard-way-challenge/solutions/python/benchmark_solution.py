"""Correctness and benchmark runner for the complete solution kernels."""

from __future__ import annotations

import argparse
import time
from typing import Callable

import torch

from solution_extension_loader import create_solution_extension


def _time(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000.0 / iters


def _make_registry(ext, dtype: torch.dtype, include_cutlass: bool):
    if dtype == torch.float32:
        registry = {
            "torch": lambda a, b: torch.matmul(a, b),
            "01_naive": lambda a, b: _fp32_out(ext.sgemm_naive, a, b),
            "02_coalesced": lambda a, b: _fp32_out(ext.sgemm_global_mem_coalesce, a, b),
            "03_shared_mem": lambda a, b: _fp32_out(ext.sgemm_shared_mem, a, b),
            "04_blocktiling_1d": lambda a, b: _fp32_out(ext.sgemm_blocktiling_1d, a, b),
            "05_blocktiling_2d": lambda a, b: _fp32_out(ext.sgemm_blocktiling_2d, a, b),
            "06_vectorize": lambda a, b: _fp32_out(ext.sgemm_vectorize, a, b),
            "07_warptiling": lambda a, b: _same_dtype_out(ext.sgemm_warptiling_default, a, b),
        }
        if include_cutlass:
            registry["13_cutlass_fp32"] = lambda a, b: _fp32_out(ext.sgemm_cutlass_fp32, a, b)
        return registry

    if dtype == torch.float16:
        registry = {
            "torch": lambda a, b: torch.matmul(a, b),
            "08_warptiling_fp16": lambda a, b: _same_dtype_out(ext.sgemm_warptiling_fp16, a, b),
            "09_tensorcore_naive_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_naive_fp16, a, b),
            "10_tensorcore_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_fp16, a, b),
            "11_tensorcore_db_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_double_buffered_fp16, a, b),
            "12_tensorcore_async_fp16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_async_fp16, a, b),
        }
        if include_cutlass:
            registry["13_cutlass_fp16"] = lambda a, b: _fp32_then_cast(ext.sgemm_cutlass_fp16, a, b)
            registry["14_cutlass_autotune_fp16_cfg0"] = lambda a, b: _fp32_then_cast_config(
                ext.sgemm_cutlass_autotune_fp16, 0, a, b
            )
        return registry

    registry = {
        "torch": lambda a, b: torch.matmul(a, b),
        "08_warptiling_bf16": lambda a, b: _same_dtype_out(ext.sgemm_warptiling_bf16, a, b),
        "09_tensorcore_naive_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_naive_bf16, a, b),
        "10_tensorcore_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_bf16, a, b),
        "11_tensorcore_db_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_double_buffered_bf16, a, b),
        "12_tensorcore_async_bf16": lambda a, b: _fp32_then_cast(ext.sgemm_tensorcore_async_bf16, a, b),
    }
    if include_cutlass:
        registry["13_cutlass_bf16"] = lambda a, b: _fp32_then_cast(ext.sgemm_cutlass_bf16, a, b)
        registry["14_cutlass_autotune_bf16_cfg0"] = lambda a, b: _fp32_then_cast_config(
            ext.sgemm_cutlass_autotune_bf16, 0, a, b
        )
    return registry


def _fp32_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(a, b, c, 1.0, 0.0)
    return c


def _same_dtype_out(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    kernel(a, b, c, 1.0, 0.0)
    return c


def _fp32_then_cast(kernel, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(a, b, c, 1.0, 0.0)
    return c.to(a.dtype)


def _fp32_then_cast_config(kernel, config_id: int, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float32)
    kernel(config_id, a, b, c, 1.0, 0.0)
    return c.to(a.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--kernels", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--include-cutlass", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    ext = create_solution_extension(include_cutlass=args.include_cutlass, verbose=args.verbose_build)
    registry = _make_registry(ext, dtype, args.include_cutlass)
    selected = args.kernels or list(registry)

    print("| kernel | size | ms | TFLOPS | max_error |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for size in args.sizes:
        a = torch.randn((size, size), device="cuda", dtype=dtype)
        b = torch.randn((size, size), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b)
        for name in selected:
            if name not in registry:
                raise ValueError(f"Unknown kernel for dtype={args.dtype}: {name}")
            out, ms = _time(lambda name=name: registry[name](a, b), args.warmup, args.iters)
            err = (out.float() - ref.float()).abs().max().item()
            ops = 2.0 * size * size * size
            print(f"| {name} | {size} | {ms:.4f} | {ops / (ms * 1e9):.2f} | {err:.4e} |")


if __name__ == "__main__":
    main()
