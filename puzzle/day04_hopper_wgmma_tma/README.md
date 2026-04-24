# Day 4 — Hopper WGMMA + TMA LOAD

**Goal:** Use the Tensor Memory Accelerator (TMA) to load A and B tiles from global memory into shared memory, replacing the generic TiledCopy from day 3.

## Background

TMA is a dedicated copy engine introduced on SM90. Instead of a cooperative per-thread copy, one thread in a block issues a single `cp.async.bulk.tensor` instruction and TMA walks a pre-built descriptor. The descriptor is constructed on the host by `make_tma_atom(SM90_TMA_LOAD{}, gmem_tensor, smem_layout, tile_shape)` and passed through to the kernel. See Part 4 of the blog.

## Task

Open `puzzle.cu` and implement the block marked `// TODO:` inside the host function `gemm_tn`. You are targeting:
- Global memory tensors `mA` / `mB` the TMA can inspect for shape and stride
- `Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM, bK))`
- `Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN, bK))`

(The `gemm_nt` host path is already filled in — use it as a reference.)

## Build & run

```bash
cmake --build build --target day04_puzzle
./build/puzzle/day04_hopper_wgmma_tma/day04_puzzle
cmake --build build --target day04_solution
./build/puzzle/day04_hopper_wgmma_tma/day04_solution
```

## Concepts you should walk away understanding

- TMA descriptors are built on the host from (gmem tensor, smem tile layout, tile shape)
- The descriptor encodes shape, stride, swizzle, OOB behavior — baked in once, reused every tile
- Why TMA wants the SMEM tile layout (for swizzle/box alignment) as well as the gmem tensor
- The kernel only sees an opaque `Copy_Atom` — all TMA inspection happens on the host
