# Day 5 — Blackwell `tcgen05.mma` + TMEM

**Goal:** Issue SM100 tcgen05 MMAs whose accumulator lives in Tensor Memory (TMEM), not registers.

## Background

Blackwell (SM100) introduces *tensor memory* (TMEM), a per-SM scratchpad that holds MMA accumulators. `tcgen05.mma` is single-issuer (one warp kicks it off) and the accumulator never materializes in registers until the epilogue copies TMEM->rmem. TMEM must be explicitly allocated by one warp with `TMEM::Allocator1Sm`, and completion is signalled via a Hopper-style barrier that MMAs `umma_arrive` on. See Part 5 of the blog.

## Task

Open `puzzle.cu` and implement the two blocks marked `// TODO:`:
1. Allocate TMEM columns with `Allocator1Sm`, and retarget `tCtAcc.data()` at `shared_storage.tmem_base_ptr`.
2. Inside the `k_tile` loop, have the elected warp issue `gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCtAcc)` for every k_block, call `umma_arrive`, and have all threads `wait_barrier` + flip the phase bit.

## Build & run

```bash
cmake --build build --target day05_puzzle
./build/puzzle/day05_blackwell_tcgen05/day05_puzzle
cmake --build build --target day05_solution
./build/puzzle/day05_blackwell_tcgen05/day05_solution
```

## Concepts you should walk away understanding

- TMEM is a named physical resource: you `allocate` it and deallocate in the epilogue
- Single-issuer semantics of tcgen05: one warp kicks off the MMA; the rest wait on a barrier
- `UMMA::ScaleOut::Zero -> One` pattern to clear the accumulator on the first MMA only
- Hopper-style phase-bit barriers (`umma_arrive` + `wait_barrier`, flip every tile)
- Why the accumulator fragment `tCtAcc` is rebound to TMEM after allocation
