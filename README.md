# cutlass-puzzle

> Learn CUTLASS and CuTe the way you learned LeetCode — one bite-sized puzzle at a time.

A hands-on companion to the blog post [*CUTLASS Deep Dive: From CuTe Layouts to
Blackwell tcgen05*](https://supercharleszhu.github.io). Every concept in the
post is re-packaged here as a compilable mini-program with the critical
block replaced by `// TODO:` — your job is to fill it in.

```
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│  Day 1: Layouts    │ ─► │  Day 2: SM80 GEMM  │ ─► │  Day 3: WGMMA      │
└────────────────────┘    └────────────────────┘    └────────────────────┘
                                                            │
                                                            ▼
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│ Day 5: tcgen05     │ ◄─ │ Day 4: WGMMA + TMA │    │ Day 4 pipelines    │
└────────────────────┘    └────────────────────┘    └────────────────────┘
         │
         ▼
┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
│ Day 6: TMA Load    │ ─► │ Day 7: Multicast   │ ─► │ Day 8: 2SM MMA     │
└────────────────────┘    └────────────────────┘    └────────────────────┘
                                                            │
                                                            ▼
                                                    ┌────────────────────┐
                                                    │ Day 9: TMA Epilog  │
                                                    └────────────────────┘
```

## Why

Because reading CUTLASS code and *writing* CUTLASS code are two very different
skills. The CUTLASS codebase is a world-class reference, but it's also ~400k
lines of heavily-templated C++ — reading it cold is like learning French by
opening Proust. This repo breaks the learning curve into 9 self-contained
puzzles you can do in one or two pomodoros each.

## Repository layout

```
cutlass-puzzle/
├── README.md                       ← you are here
├── CMakeLists.txt                  ← top-level build
├── cmake/                          ← reserved for future FindXxx.cmake
├── blog/                           ← full, unmodified tutorial code
│   ├── 01_tiled_copy.cu
│   ├── 02_sgemm_sm80.cu
│   ├── 03_wgmma_sm90.cu
│   ├── 04_wgmma_tma_sm90.cu
│   ├── 05_blackwell_mma_sm100.cu
│   ├── 06_blackwell_mma_tma_sm100.cu
│   ├── 07_blackwell_mma_tma_multicast_sm100.cu
│   ├── 08_blackwell_mma_tma_2sm_sm100.cu
│   └── 09_blackwell_mma_tma_epi_sm100.cu
└── puzzle/                         ← the challenges
    ├── day01_layouts_and_tiled_copy/
    │   ├── puzzle.cu               ← TODOs to fill in
    │   ├── solution.cu             ← reference answer
    │   └── README.md               ← task + hints
    ├── day02_sm80_pipelined_sgemm/
    ├── day03_hopper_wgmma/
    ├── day04_hopper_wgmma_tma/
    ├── day05_blackwell_tcgen05/
    ├── day06_blackwell_tma_load/
    ├── day07_blackwell_tma_multicast/
    ├── day08_blackwell_2sm_mma/
    └── day09_blackwell_tma_epilogue/
```

## The puzzles

| Day | Topic                                    | Min GPU         | Time |
|-----|------------------------------------------|-----------------|------|
| 1   | Layouts, `TiledCopy`, vectorized loads   | any             | 20m  |
| 2   | SM80 software-pipelined SGEMM            | A100 (sm_80)    | 1h   |
| 3   | Hopper WGMMA async dance                 | H100 (sm_90a)   | 30m  |
| 4   | Hopper TMA + WGMMA producer/consumer     | H100 (sm_90a)   | 1h   |
| 5   | Blackwell `tcgen05.mma` + TMEM alloc     | B200 (sm_100a)  | 1h   |
| 6   | Blackwell TMA load                       | B200 (sm_100a)  | 30m  |
| 7   | Blackwell TMA multicast                  | B200 (sm_100a)  | 1h   |
| 8   | Blackwell 2SM MMA (256×256)              | B200 (sm_100a)  | 1h   |
| 9   | Blackwell TMA epilogue                   | B200 (sm_100a)  | 1h   |

No hardware? The code still *compiles* with `nvcc -arch=sm_90a` / `sm_100a`
and you can verify your solution with `diff` against `solution.cu`.

## Dependencies

| Dep                     | Version                  | Notes                                           |
|-------------------------|--------------------------|-------------------------------------------------|
| CUDA Toolkit            | **12.4+** (12.8 for SM100) | Provides `nvcc`, CUDA headers, cuBLAS           |
| NVIDIA driver           | ≥ 550 for SM90, ≥ 570 for SM100 | Runtime device support                 |
| CMake                   | **3.18+**                | Uses `CUDA_ARCHITECTURES` target property       |
| Ninja (recommended)     | 1.10+                    | Speeds up CuTe template compilation             |
| GCC / Clang             | C++17 capable (GCC 9+)   | Compiles host code                              |
| [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) | **3.6+** / 4.x | Source tree; header-only                       |
| thrust                  | bundled with CUDA        | For host/device vectors in examples             |

A hint about CUTLASS versions: the Blackwell tutorials (days 5–9) require a
CUTLASS release that includes `cute/arch/tmem_allocator_sm100.hpp` — this
landed in CUTLASS 3.6 and has been iterated on through 4.x.

## Setup

### 1. Clone with the CUTLASS submodule

```bash
git clone --recursive https://github.com/supercharleszhu/cutlass-puzzle.git
cd cutlass-puzzle
```

Or if you already cloned without `--recursive`:

```bash
git submodule add https://github.com/NVIDIA/cutlass third_party/cutlass
```

Or point at an existing CUTLASS checkout:

```bash
export CUTLASS_DIR=/path/to/your/cutlass
```

### 2. Configure

```bash
# Default (SM80 only)
cmake -B build -GNinja

# Multiple archs
cmake -B build -GNinja -DCUTLASS_PUZZLE_CUDA_ARCHS="80;90a;100a"

# Release build (strongly recommended — Debug builds of CuTe code are slow)
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_PUZZLE_CUDA_ARCHS="90a"
```

> **arch suffixes**: CUTLASS's WGMMA/tcgen05 code requires the `a` suffix
> (`90a`, `100a`, etc.) — this enables the "architecture-specific" PTX
> extensions.

### 3. Build

```bash
# One puzzle
cmake --build build --target day01_puzzle

# All puzzles + all solutions
cmake --build build --target puzzle_all

# All blog reference files
cmake --build build --target blog_all
```

### 4. Run

Executables land in `build/puzzle/dayNN_xxx/` and `build/blog/`:

```bash
./build/puzzle/day01_layouts_and_tiled_copy/day01_puzzle    # will FAIL verification
# fix puzzle.cu
./build/puzzle/day01_layouts_and_tiled_copy/day01_puzzle    # now prints "Success."
./build/puzzle/day01_layouts_and_tiled_copy/day01_solution  # for comparison
```

## How to use this repo

1. Open the day's `README.md` and skim the task.
2. Open `puzzle.cu`, find the `// TODO` block.
3. Try. Think in terms of *layouts and atoms*, not raw threads and indices.
4. Build. If it compiles but verifies wrong, you have a logic bug.
5. If stuck for more than 20 min, peek at `solution.cu` — this is for learning,
   not gatekeeping.
6. Read the corresponding section of the [blog post](https://supercharleszhu.github.io)
   for deeper context.

## Learning path recommendation

**If you know CUDA but not CuTe:** days 1 → 2 give you 90% of what you need to
read any CuTe code.

**If you want to understand FlashAttention 3 / FlexAttention:** days 3 → 4 are
non-negotiable. WGMMA + TMA producer-consumer is the mental model.

**If you want to write production Blackwell kernels:** days 5 → 9, in order.
Each one introduces exactly one new primitive; don't skip ahead.

## References

- [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — the upstream library.
  All puzzle source is derived from `examples/cute/tutorial/`.
- [CuTe documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cpp/cute).
- [Hopper architecture whitepaper](https://resources.nvidia.com/en-us-tensor-core).
- [Blackwell architecture whitepaper](https://resources.nvidia.com/en-us-blackwell-architecture).
- Blog post: [*CUTLASS Deep Dive*](https://supercharleszhu.github.io).

## License

Source files under `blog/` and `puzzle/*/solution.cu` are copied from NVIDIA's
CUTLASS repository and retain their original **BSD-3-Clause** license (see
header comments in each file). Everything else in this repo is BSD-3-Clause
as well — see [LICENSE](LICENSE).

## Contributing

Bug reports and new puzzle ideas welcome — open an issue. Particularly useful:

- Cleaner puzzle scoping (blanking the *right* lines).
- Additional days: FP8 blockwise scaling, grouped GEMM, EVT epilogue.
- CI that builds against multiple CUTLASS versions.
