# Puzzle

9 progressively harder days of CUTLASS/CuTe challenges, each a self-contained
buildable `.cu` program. Every day has:

| File          | What it is                                                     |
|---------------|----------------------------------------------------------------|
| `puzzle.cu`   | Working scaffold with critical blocks replaced by `// TODO`    |
| `solution.cu` | The full reference — peek only after you've tried              |
| `README.md`   | Task statement, hints, build commands                          |

## The nine days

| Day | Focus                                       | Minimum GPU    |
|-----|---------------------------------------------|----------------|
| 1   | Layouts + vectorized `TiledCopy`            | Any CUDA GPU   |
| 2   | SM80 software-pipelined SGEMM               | SM80 (A100)    |
| 3   | Hopper WGMMA (async MMA dance)              | SM90 (H100)    |
| 4   | Hopper WGMMA + TMA atoms                    | SM90 (H100)    |
| 5   | Blackwell `tcgen05.mma` + TMEM              | SM100 (B200)   |
| 6   | Blackwell + TMA load                        | SM100          |
| 7   | Blackwell TMA multicast across a cluster    | SM100          |
| 8   | Blackwell 2SM MMA (256×256)                 | SM100          |
| 9   | Blackwell TMA epilogue                      | SM100          |

Each day builds on the previous. If you can't run SM90/SM100 hardware,
you can still compile most of them with `-arch=sm_90a` / `-arch=sm_100a`
via `nvcc` and verify with static analysis.

## How to work a puzzle

```bash
# Build one day
cmake --build build --target day03_puzzle

# Run
./build/puzzle/day03_puzzle

# It will fail verification until you fix the TODOs.
# Once passing, build and diff against the reference:
cmake --build build --target day03_solution
diff day03_hopper_wgmma/puzzle.cu day03_hopper_wgmma/solution.cu
```

## Build all puzzles

```bash
cmake --build build --target puzzle_all
```

This builds every `dayNN_puzzle` and `dayNN_solution` whose minimum compute
capability is reachable by your configured `CUTLASS_PUZZLE_CUDA_ARCHS`.
