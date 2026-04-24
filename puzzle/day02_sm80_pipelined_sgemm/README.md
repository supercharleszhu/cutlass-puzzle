# Day 2 — SM80 Software-Pipelined SGEMM

**Goal:** Build a 3-stage software pipeline that overlaps gmem->smem copies, smem->rmem loads, and register-level MMAs on Ampere.

## Background

On SM80, `cp.async` gives you asynchronous gmem->smem loads that retire in program order via `cp.async.commit_group` / `cp.async.wait_group`. To hide both gmem latency and smem latency, you run three things at once: a gmem->smem pipeline (K_PIPE_MAX-deep), a smem->rmem double buffer inside each k-tile (via ldmatrix), and the MMAs on the previous register buffer. See Part 2 of the blog.

## Task

Open `puzzle.cu` and implement the block marked `// TODO:`. You are targeting:
- Prologue that issues `K_PIPE_MAX - 1` cp.async-fenced copies of A/B tiles into smem
- A `CUTE_NO_UNROLL while` loop over `k_tile_count > -(K_PIPE_MAX-1)` that inside a `CUTE_UNROLL for (k_block...)` performs the smem->rmem prefetch of k_block+1, issues a fresh cp.async at k_block==0, and calls `gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC)`
- Correct cursor arithmetic for `smem_pipe_read` and `smem_pipe_write`

## Build & run

```bash
cmake --build build --target day02_puzzle
./build/puzzle/day02_sm80_pipelined_sgemm/day02_puzzle
cmake --build build --target day02_solution
./build/puzzle/day02_sm80_pipelined_sgemm/day02_solution
```

## Concepts you should walk away understanding

- How `cp.async` / `cp_async_fence` / `cp_async_wait<N>` implement a multi-stage async pipeline
- Why you need both a gmem->smem pipeline AND a smem->rmem register pipeline to hit peak throughput
- Why `cp_async_wait<K_PIPE_MAX-2>` (not `wait<0>`) is the correct barrier depth
- The role of `smem_pipe_read`/`smem_pipe_write` as rotating indices into a ring buffer
- How the loop tail drains (the `-(K_PIPE_MAX-1)` guard) without issuing new loads
