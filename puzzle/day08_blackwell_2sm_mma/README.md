# Day 8 — Blackwell 2SM MMA

**Goal:** Promote the day-7 single-SM tcgen05 MMA into a 2SM MMA whose accumulator is shared across a pair of peer CTAs, and route leader-only responsibilities correctly.

## Background

Blackwell's `tcgen05.mma` has a 2SM variant that pairs two CTAs along a cluster axis and produces a larger (256x256) output tile. One CTA in the pair is the "leader" — it owns the MMA barrier and advertises TMA transaction bytes. The other is the "peer" — it still issues its half of the TMA load, but does not wait. See Part 8 of the blog.

## Task

Open `puzzle.cu` and implement the three blocks marked `// TODO:`:
1. Swap the 1SM atom for `SM100_MMA_F16BF16_2x1SM_SS<TypeA, TypeB, TypeC, 256, 256, UMMA::Major::K, UMMA::Major::K>` in `make_tiled_mma`.
2. Define `elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{}`.
3. Guard `set_barrier_transaction_bytes` behind `if (elect_one_cta)` so only the leader announces expected bytes.

## Build & run

```bash
cmake --build build --target day08_puzzle
./build/puzzle/day08_blackwell_2sm_mma/day08_puzzle
cmake --build build --target day08_solution
./build/puzzle/day08_blackwell_2sm_mma/day08_solution
```

## Concepts you should walk away understanding

- 2SM MMAs widen the logical tile (128 -> 256 along M) by splitting the accumulator across a pair of peer CTAs' TMEM
- Leader / peer asymmetry: both CTAs issue TMA, only the leader owns the barrier
- Why the V-axis (`get<0>(cta_in_cluster_coord_vmnk)`) encodes the in-pair role
- The `SM100_MMA_F16BF16_2x1SM_SS` naming: 2 SM, 1 CTA per instruction issuer, Shared-memory for both A and B
