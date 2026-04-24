/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

// This is a simple tutorial showing several ways to partition a tensor into tiles then
// perform efficient, coalesced copies. This example also shows how to vectorize accesses
// which may be a useful optimization or required for certain workloads.
//
// `copy_kernel()` and `copy_kernel_vectorized()` each assume a pair of tensors with
// dimensions (m, n) have been partitioned via `tiled_divide()`.
//
// The result are a part of compatible tensors with dimensions ((M, N), m', n'), where
// (M, N) denotes a statically sized tile, and m' and n' denote the number of such tiles
// within the tensor.
//
// Each statically sized tile is mapped to a CUDA threadblock which performs efficient
// loads and stores to Global Memory.
//
// `copy_kernel()` uses `cute::local_partition()` to partition the tensor and map
// the result to threads using a striped indexing scheme. Threads themselve are arranged
// in a (ThreadShape_M, ThreadShape_N) arrangement which is replicated over the tile.
//
// `copy_kernel_vectorized()` uses `cute::make_tiled_copy()` to perform a similar
// partitioning using `cute::Copy_Atom` to perform vectorization. The actual vector
// size is defined by `ThreadShape`.
//
// This example assumes the overall tensor shape is divisible by the tile size and
// does not perform predication.


/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template <class TensorS, class TensorD, class ThreadLayout>
__global__ void copy_kernel(TensorS S, TensorD D, ThreadLayout)
{
  using namespace cute;

  // Slice the tiled tensors
  Tensor tile_S = S(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_,_), blockIdx.x, blockIdx.y);   // (BlockShape_M, BlockShape_N)

  // Construct a partitioning of the tile among threads with the given thread arrangement.

  // Concept:                         Tensor  ThrLayout       ThrIndex
  Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)
  Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);  // (ThrValM, ThrValN)

  // Construct a register-backed Tensor with the same shape as each thread's partition
  // Use make_tensor to try to match the layout of thr_tile_S
  Tensor fragment = make_tensor_like(thr_tile_S);               // (ThrValM, ThrValN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(thr_tile_S, fragment);
  copy(fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
  using namespace cute;

  // Slice the tensors to obtain a view into each tile.
  Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)
  Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y);  // (BlockShape_M, BlockShape_N)

  // Construct a Tensor corresponding to each thread's slice.
  ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

  Tensor thr_tile_S = thr_copy.partition_S(tile_S);             // (CopyOp, CopyM, CopyN)
  Tensor thr_tile_D = thr_copy.partition_D(tile_D);             // (CopyOp, CopyM, CopyN)

  // Construct a register-backed Tensor with the same shape as each thread's partition
  // Use make_fragment because the first mode is the instruction-local mode
  Tensor fragment = make_fragment_like(thr_tile_D);             // (CopyOp, CopyM, CopyN)

  // Copy from GMEM to RMEM and from RMEM to GMEM
  copy(tiled_copy, thr_tile_S, fragment);
  copy(tiled_copy, fragment, thr_tile_D);
}

/// Main function
int main(int argc, char** argv)
{
  //
  // Given a 2D shape, perform an efficient copy
  //

  using namespace cute;
  using Element = float;

  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(256, 512);

  //
  // Allocate and initialize
  //

  thrust::host_vector<Element> h_S(size(tensor_shape));
  thrust::host_vector<Element> h_D(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
    h_D[i] = Element{};
  }

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  //
  // Make tensors
  //
  // NOTE: make_tensor() does NOT copy data. It's a zero-cost view — a lightweight
  // wrapper pairing an existing pointer with layout metadata. No allocation, no
  // bytes move. Mutating tensor_S(i, j) mutates the d_S buffer directly.
  // Analogy: std::span, not std::vector.
  //
  // make_gmem_ptr / make_smem_ptr / make_rmem_ptr are also free — they just tag
  // the raw pointer with a memory-space type so CuTe can later emit the right
  // ld.global / ld.shared / register-file load instruction.

  Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
  Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));

  //
  // Tile tensors
  //

  // Define a statically sized block (M, N).
  // Note, by convention, capital letters are used to represent static modes.
  auto block_shape = make_shape(Int<128>{}, Int<64>{});

  if ((size<0>(tensor_shape) % size<0>(block_shape)) || (size<1>(tensor_shape) % size<1>(block_shape))) {
    std::cerr << "The tensor shape must be divisible by the block shape." << std::endl;
    return -1;
  }
  // Equivalent check to the above
  if (not evenly_divides(tensor_shape, block_shape)) {
    std::cerr << "Expected the block_shape to evenly divide the tensor shape." << std::endl;
    return -1;
  }

  // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
  // shape, and modes (m', n') correspond to the number of tiles.
  //
  // These will be used to determine the CUDA kernel grid dimensions.
  Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape);      // ((M, N), m', n')
  Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);      // ((M, N), m', n')

  // =============================================================================
  // PUZZLE DAY 1 — TODO: Construct a TiledCopy for this 128x64 float tile.
  //
  // Target: 256 threads arranged 32x8, each issuing one 128-bit vectorized
  //         load of 4 contiguous floats. This tiles the (128, 64) block
  //         cleanly (32*4 = 128 in M, 8*1 = 8 in N with each thread looping
  //         over the remaining 8 columns).
  //
  // You need to fill in:
  //   - thr_layout: a Layout with shape (32, 8) mapping linear threadIdx.x
  //                 to a 2D (m, n) thread coordinate.
  //   - val_layout: a Layout with shape (4, 1) — the per-thread value tile.
  //   - CopyOp:     UniversalCopy<uint_byte_t<...>> where the byte width
  //                 equals sizeof(Element) * size(val_layout) (= 16 bytes).
  //
  // Once the three declarations are correct, `make_tiled_copy` and the
  // kernel launch below work as-is. Compile and run — verification at
  // the bottom of main() will print "Success." if correct.
  //
  // If you get stuck, see Part 2 of the blog post and `solution.cu`.
  // =============================================================================

  Layout thr_layout = /* TODO */ make_layout(make_shape(Int<1>{}, Int<1>{}));
  Layout val_layout = /* TODO */ make_layout(make_shape(Int<1>{}, Int<1>{}));

  using CopyOp = /* TODO */ AutoVectorizingCopy;
  using Atom   = Copy_Atom<CopyOp, Element>;

  TiledCopy tiled_copy = make_tiled_copy(Atom{}, thr_layout, val_layout);

  // These asserts enforce the pedagogical goal. They will fire until your
  // thr_layout and val_layout together cover the (128, 64) block_shape AND
  // each thread issues a 4-element (16-byte) vector.
  static_assert(size(thr_layout) * size(val_layout) == size(block_shape),
      "PUZZLE: thr_layout * val_layout must cover the whole block tile.");
  static_assert(size(thr_layout) == 256,
      "PUZZLE: target 256 threads per block (32 x 8).");
  static_assert(size(val_layout) == 4,
      "PUZZLE: target 4 values per thread (one 128-bit vector).");
  static_assert(!std::is_same_v<CopyOp, AutoVectorizingCopy>,
      "PUZZLE: use UniversalCopy<uint_byte_t<sizeof(Element)*size(val_layout)>>.");

  //
  // Determine grid and block dimensions
  //

  dim3 gridDim (size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));   // Grid shape corresponds to modes m' and n'
  dim3 blockDim(size(thr_layout));

  //
  // Launch the kernel
  //
  copy_kernel_vectorized<<< gridDim, blockDim >>>(
    tiled_tensor_S,
    tiled_tensor_D,
    tiled_copy);

  cudaError result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
    return -1;
  }

  //
  // Verify
  //

  h_D = d_D;

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}

