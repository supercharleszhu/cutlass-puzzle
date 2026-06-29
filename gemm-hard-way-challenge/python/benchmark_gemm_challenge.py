"""Minimal benchmark scaffold for the standalone GEMM challenge.

This file intentionally does not auto-build the upstream extension. Fill
`load_extension()` and register kernels as you complete each CUDA file.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class BenchResult:
    name: str
    size: int
    ms: float
    tflops: float
    max_error: float


def load_extension():
    # TODO(student): load your completed CUDA extension with
    # torch.utils.cpp_extension.load or your preferred build system.
    return None


def register_kernels(ext) -> dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    kernels: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
        "torch": lambda a, b: torch.matmul(a, b),
    }

    # TODO(student): uncomment/add entries as files become correct.
    # kernels["01_naive"] = lambda a, b: ext.sgemm_naive(a, b)
    # kernels["02_coalesced"] = lambda a, b: ext.sgemm_global_mem_coalesce(a, b)
    # kernels["03_shared_mem"] = lambda a, b: ext.sgemm_shared_mem(a, b)

    return kernels


def bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> tuple[torch.Tensor, float]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()

    return out, (time.perf_counter() - start) * 1000.0 / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    ext = load_extension()
    kernels = register_kernels(ext)

    print("| kernel | size | ms | TFLOPS | max error |")
    print("| --- | ---: | ---: | ---: | ---: |")

    for size in args.sizes:
        a = torch.randn((size, size), device="cuda", dtype=dtype)
        b = torch.randn((size, size), device="cuda", dtype=dtype)
        ref = torch.matmul(a, b)

        for name, fn in kernels.items():
            out, ms = bench(lambda: fn(a, b), args.warmup, args.iters)
            ops = 2.0 * size * size * size
            err = (out.float() - ref.float()).abs().max().item()
            print(f"| {name} | {size} | {ms:.4f} | {ops / (ms * 1e9):.2f} | {err:.3e} |")


if __name__ == "__main__":
    main()

