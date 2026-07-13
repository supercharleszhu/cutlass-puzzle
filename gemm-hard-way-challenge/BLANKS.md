# GEMM Challenge Blanks

Use this as your checklist. Each item corresponds to a `GEMM_TODO_*` placeholder in `cuda/`.

| File | Fill-in focus |
| --- | --- |
| `01_naive.cu` | Non-coalesced row/column mapping, dot-product load, alpha/beta epilogue. |
| `02_kernel_global_mem_coalesce.cu` | Coalesced row/column mapping for warp lanes. |
| `03_kernel_shared_mem.cu` | A/B global-to-shared loads and shared-memory dot product. |
| `04_kernel_blocktiling_1d.cu` | Tile loads, B register cache, TM per-thread output update. |
| `05_kernel_blocktiling_2d.cu` | A/B tile loads, register fragments, TMxTN outer-product FMA. |
| `06_kernel_vectorize.cu` | `float4` A/B loads and shared-to-register loads. |
| `07_kernel_warptiling.cu` | Warp subtile constants and warp-fragment loads. |
| `08_kernel_warptiling_all_dtypes.cu` | Dtype-aware accumulator initialization and type reasoning. |
| `09_kernel_tensorcore_naive.cu` | WMMA tile mapping, fragment loads, and `mma_sync`. |
| `10_kernel_tensorcore_warptiled.cu` | WMMA loads/MMA from tiled shared-memory fragments. |
| `11_kernel_tensorcore_double_buffered.cu` | Read/write buffer selection and buffer-flip pipeline. |
| `12_kernel_tensorcore_async.cu` | Async pipeline compute placeholder. |
| `13_kernel_cutlass.cu` | CUTLASS `GemmShape<BM, BN, BK>`: `BM` rows of C/A per CTA, `BN` cols of C/B per CTA, `BK` reduction depth per mainloop stage. Start with `128,128,32`; warp shape `64,64,32`; instruction shape `16,8,16`. |
| `14_kernel_cutlass_autotunable.cu` | First autotune config entry and search-space extension. |
| `15_kernel_cutlass_hopper.cu` | CUTLASS official SM90 schedule selection: TMA warp-specialized, ping-pong, cooperative/persistent, Stream-K. |
| `16_kernel_cutlass_hopper_autotunable.cu` | Runtime autotune knobs: tile shape, raster order, decomposition mode, swizzle, split count. |
| `17_kernel_hopper_tma_wgmma.cu` | Blog kernel 2 analogue: BF16 TMA + WGMMA baseline via `GemmUniversal`. |
| `18_kernel_fastcu_matmul2_manual_tma_wgmma.cu` | fast.cu matmul_2 analogue: manually apply 2D TMA loads, barriers, WGMMA m64n64k16, and C^T stores. |
| `19_kernel_hopper_fastcu_big_tile.cu` | Blog kernel 3/5 analogue: 128x256x64 tile and WGMMA/register-pressure tradeoff. |
| `20_kernel_hopper_fastcu_persistent.cu` | Blog kernel 6 analogue: persistent scheduling over the output tile space. |
| `21_kernel_hopper_fastcu_cluster.cu` | Blog kernel 8 analogue: 2x1 CTA cluster and CUTLASS cooperative SM90 scheduling. |
| `22_kernel_fastcu_handwritten_tma_wgmma.cu` | Blog kernel 12 analogue: handwritten WGMMA/TMA, TMA store, Hilbert scheduling, and B^T/C^T layout bridge. |
| `23_kernel_fastcu_cached_tma_maps.cu` | fast.cu benchmark assumption: stable buffers and cached TMA tensor maps. |
| `24_kernel_fastcu_final.cu` | final benchmark day: compare 4096/8192 TFLOPS to fast.cu README. |
| `25_kernel_fastcu_tma_store.cu` | epilogue day: accumulator conversion, stmatrix, shared layout padding, and TMA store. |
| `26_kernel_fastcu_hilbert_final.cu` | scheduling day: Hilbert tile order, persistent CTAs, and final performance analysis. |
