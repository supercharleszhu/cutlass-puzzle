# Day 1 — Layouts & Tiled Copy

**Goal:** implement a coalesced, vectorized copy of a `(256, 512)` float tensor
by constructing a `TiledCopy` from scratch.

## Background

A *Layout* is a function from coordinates → offsets. A *TiledCopy* composes:

1. A **thread layout** — which thread owns which `(m, n)` position.
2. A **value layout** — how many contiguous elements each thread handles per call.
3. A **Copy atom** — the PTX-level instruction emitted (`ld.global.v4.f32`, etc.).

If you get the combination right, 32 threads in a warp fire a single 128-byte
global memory transaction, and the memory subsystem is happy.

## Task

Open `puzzle.cu` and fill in the three `// TODO` lines marked in the source.
You are targeting:

- **256 threads** per CTA arranged as 32 × 8 (M × N).
- Each thread loads **4 contiguous floats** in M.
- One 128-bit (16-byte) vector instruction per thread.

## Build & run

```bash
# From repo root
cmake --build build --target day01_puzzle
./build/puzzle/day01_layouts_and_tiled_copy/day01_puzzle
# Expected: "Success."

# Compare against reference
cmake --build build --target day01_solution
./build/puzzle/day01_layouts_and_tiled_copy/day01_solution
```

## What to check

- `nvcc` must accept the types — `make_tiled_copy` will fail compilation if
  `thr_layout * val_layout` does not match the tile shape.
- The kernel launch uses `size(thr_layout)` as the block dim, so a wrong
  thread count will cause a wrong result rather than a compile error.
- Try `print(tiled_copy)` inside `main()` to see the layout CuTe built.

## Concepts you should walk away understanding

- What `make_layout(make_shape(Int<32>{}, Int<8>{}))` *is* as a function.
- Why the thread layout's M mode is 32 (one warp worth).
- Why vectorizing from 1 float → 4 floats reduces memory transactions by 4×.
- The relationship between `size(thr_layout) * size(val_layout)` and the
  tile shape `(BlockM, BlockN)`.
