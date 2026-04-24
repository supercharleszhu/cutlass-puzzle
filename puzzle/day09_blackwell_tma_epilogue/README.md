# Day 9 — Blackwell TMA Epilogue

**Goal:** Build the full axpby epilogue pipeline: TMA load C, read accumulator from TMEM, fuse `D = alpha*A*B + beta*C`, and TMA-store D back to gmem.

## Background

The epilogue tile-loops over epi tiles: each iteration pulls one slab of C from gmem via TMA into smem, loads C smem->rmem, loads the matching accumulator slab from TMEM->rmem, fuses `axpby`, stores D rmem->smem, and launches a TMA store smem->gmem. The `tma_store_fence` + `tma_store_arrive` + `tma_store_wait` sequence is the store-side analogue of the load barriers from day 6. See Part 9 of the blog.

## Task

Open `puzzle.cu` and fill in the `// TODO:` body inside the `for (int epi_tile_idx ...)` loop:
- Single-thread TMA load of C via `tma_atom_C.with(barrier, 0)` and `set_barrier_transaction_bytes`
- `wait_barrier` + phase-bit flip
- `copy_aligned(tTR_sC, tTR_rC)` smem->rmem
- `copy(t2r_copy, tTR_tAcc(_,_,epi_tile_idx), tTR_rD)` TMEM->rmem
- `axpby(beta, tTR_rC, alpha, tTR_rD)`
- `copy_aligned(tTR_rD, tTR_sD)` rmem->smem
- `tma_store_fence`, single-thread `copy(tma_atom_D, tSG_sD, tSG_gD(_,epi_tile_idx))`, `tma_store_arrive`, `tma_store_wait<0>`
- `__syncthreads` at the end

## Build & run

```bash
cmake --build build --target day09_puzzle
./build/puzzle/day09_blackwell_tma_epilogue/day09_puzzle
cmake --build build --target day09_solution
./build/puzzle/day09_blackwell_tma_epilogue/day09_solution
```

## Concepts you should walk away understanding

- Epilogue as an independent pipeline: TMA load C, TMEM->reg, compute, reg->smem, TMA store D
- The store-side barrier protocol: `tma_store_fence` (commit smem writes) -> `tma_store_arrive` (commit the store) -> `tma_store_wait<0>` (drain)
- Why `axpby` must be fused in rmem (C and D share no gmem buffer; TMEM only holds `A*B`)
- Why there is a `__syncthreads` between `copy_aligned(smem->rmem for C)` and `copy_aligned(rmem->smem for D)` — smem is reused for both
