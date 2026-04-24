# Day 3 — Hopper WGMMA Mainloop

**Goal:** Issue async warpgroup-level matrix multiplies (WGMMA) on SM90 using the 6-call fence/arrive/commit/wait dance.

## Background

SM90 replaces per-thread `mma.sync` with warpgroup-wide async MMAs (`wgmma.mma_async`) consumed by all 128 threads of a warpgroup. Because the MMA runs asynchronously, CUTLASS wraps each issue with explicit fences, an arrive, a commit_batch, and a wait so the compiler and hardware do not reorder operand/accumulator accesses. See Part 3 of the blog.

## Task

Open `puzzle.cu` and implement the block marked `// TODO:` inside the mainloop. You are targeting:
- `warpgroup_fence_operand(tCrC)` before the MMA, to mark the accumulator live-in
- `warpgroup_arrive()` to release operands to the MMA unit
- `cute::gemm(mma, tCrA(_,_,_,k_pipe_read), tCrB(_,_,_,k_pipe_read), tCrC)`
- `warpgroup_commit_batch()` to bundle this MMA into a group
- `warpgroup_wait<0>()` to drain all outstanding WGMMAs
- Final `warpgroup_fence_operand(tCrC)` to mark the accumulator live-out

## Build & run

```bash
cmake --build build --target day03_puzzle
./build/puzzle/day03_hopper_wgmma/day03_puzzle
cmake --build build --target day03_solution
./build/puzzle/day03_hopper_wgmma/day03_solution
```

## Concepts you should walk away understanding

- Why WGMMA is async and needs explicit fences around accumulator state
- The arrive/commit/wait split: arrive = operands are ready, commit = group them, wait<N> = drain down to N outstanding groups
- How fences block compiler reordering of register reads/writes relative to the async MMA
- Why the whole warpgroup (128 threads) participates and the accumulator tile is SIMT-distributed across it
