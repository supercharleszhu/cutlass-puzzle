# Day 7 — Blackwell TMA Multicast

**Goal:** Share A tiles along the cluster's N-axis and B tiles along the M-axis with a single multicast TMA load, and signal MMA completion to every multicast participant.

## Background

In a Hopper/Blackwell cluster, multiple CTAs within the same cluster often need identical A or B tiles. TMA multicast lets one CTA's issue fan out the load into N or M neighbors' SMEM. The "which neighbors" is encoded in a `uint16_t` mask computed by `create_tma_multicast_mask<axis...>(cluster_layout_vmnk, cta_in_cluster_coord_vmnk)`. See Part 7 of the blog.

## Task

Open `puzzle.cu` and implement the two blocks marked `// TODO:`:
1. Compute `tma_mcast_mask_a` (axis=2, N-mode), `tma_mcast_mask_b` (axis=1, M-mode), and `mma_mcast_mask_c` as the OR of `<0,1>` (VM) and `<0,2>` (VN) projections.
2. Replace the single-CTA `umma_arrive` with `umma_arrive_multicast(&mma_barrier, mma_mcast_mask_c)` so all multicast participants observe MMA completion.

## Build & run

```bash
cmake --build build --target day07_puzzle
./build/puzzle/day07_blackwell_tma_multicast/day07_puzzle
cmake --build build --target day07_solution
./build/puzzle/day07_blackwell_tma_multicast/day07_solution
```

## Concepts you should walk away understanding

- TMA multicast as a bandwidth amplifier: one gmem read, many smem destinations
- Why A multicasts along N (every CTA at the same M needs the same A tile for different N) and B multicasts along M
- How `create_tma_multicast_mask<axes...>` projects the cluster layout to the sharing axes
- Why the MMA arrival must be multicast too — otherwise the CTAs that received A or B via multicast will never observe the barrier flip
