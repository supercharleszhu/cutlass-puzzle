# Day 6 — Blackwell TMA Load

**Goal:** Replace the cooperative per-thread copy from day 5 with TMA bulk loads on SM100.

## Background

Blackwell keeps SM90's TMA engine. One thread of one warp issues a `cp.async.bulk.tensor` per tile and an mbarrier tracks bytes-in-flight. The host builds a `Copy_Atom` via `make_tma_atom(SM90_TMA_LOAD{}, gmem_tensor, smem_layout, tile_shape)`, and the kernel binds the atom to a barrier with `.with(barrier)` before calling `copy(...)`. See Part 6 of the blog.

## Task

Open `puzzle.cu` and implement the two blocks marked `// TODO:`:
1. **Host side**: construct `tma_atom_A` and `tma_atom_B` from `mA` / `mB`, the SMEM layouts `sA_layout` / `sB_layout`, and the MK / NK projections of `mma_tiler`. Expose them as TMA-aware tensors via `get_tma_tensor(shape(...))`.
2. **Kernel side** (inside the `k_tile` loop): elect one thread of one warp, call `set_barrier_transaction_bytes`, then issue two `copy(tma_atom_*.with(barrier), ...)` calls.

## Build & run

```bash
cmake --build build --target day06_puzzle
./build/puzzle/day06_blackwell_tma_load/day06_puzzle
cmake --build build --target day06_solution
./build/puzzle/day06_blackwell_tma_load/day06_solution
```

## Concepts you should walk away understanding

- TMA as a descriptor-driven copy engine: one thread issues, the engine walks the descriptor
- The mbarrier "transaction bytes" protocol: set expected bytes, then the TMA updates the barrier when bytes land
- Why TMA loads must be behind an `elect_one_warp && elect_one_thr` guard
- `select<0,2>(mma_tiler)` and `select<1,2>(mma_tiler)` as MK / NK projections of the 3-D tiler
