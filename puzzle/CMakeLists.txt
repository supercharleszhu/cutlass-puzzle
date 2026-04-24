# Puzzle targets — one puzzle + one solution executable per day.
# Gated by minimum compute capability: the target is skipped if no
# requested CUDA arch meets the minimum.

set(PUZZLE_TARGETS "")

function(add_day day_dir day_label min_cc)
  set(puzzle_tgt "${day_label}_puzzle")
  set(soln_tgt   "${day_label}_solution")

  add_executable(${puzzle_tgt} ${day_dir}/puzzle.cu)
  add_executable(${soln_tgt}   ${day_dir}/solution.cu)

  foreach(t IN ITEMS ${puzzle_tgt} ${soln_tgt})
    target_link_libraries(${t} PRIVATE cutlass_puzzle_common)
    set_target_properties(${t} PROPERTIES CUDA_ARCHITECTURES "${CUTLASS_PUZZLE_CUDA_ARCHS}")
    # Headers (e.g. example_utils.hpp) live in this directory and in the day directory.
    target_include_directories(${t} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/${day_dir})
    set_property(TARGET ${t} APPEND PROPERTY COMPILE_DEFINITIONS "PUZZLE_MIN_CC=${min_cc}")
  endforeach()

  list(APPEND PUZZLE_TARGETS ${puzzle_tgt} ${soln_tgt})
  set(PUZZLE_TARGETS ${PUZZLE_TARGETS} PARENT_SCOPE)
endfunction()

add_day(day01_layouts_and_tiled_copy     day01  70)
add_day(day02_sm80_pipelined_sgemm       day02  80)
add_day(day03_hopper_wgmma               day03  90)
add_day(day04_hopper_wgmma_tma           day04  90)
add_day(day05_blackwell_tcgen05          day05 100)
add_day(day06_blackwell_tma_load         day06 100)
add_day(day07_blackwell_tma_multicast    day07 100)
add_day(day08_blackwell_2sm_mma          day08 100)
add_day(day09_blackwell_tma_epilogue     day09 100)

add_custom_target(puzzle_all DEPENDS ${PUZZLE_TARGETS})
