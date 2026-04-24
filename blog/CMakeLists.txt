# Blog examples — verbatim copies of the CUTLASS tutorials referenced in the
# blog post. Each target is gated by the minimum compute capability.

set(BLOG_TARGETS "")

function(add_blog_example name source min_cc)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE cutlass_puzzle_common)
  # Only build if any user-requested arch meets the minimum.
  set_target_properties(${name} PROPERTIES CUDA_ARCHITECTURES "${CUTLASS_PUZZLE_CUDA_ARCHS}")
  set_property(TARGET ${name} APPEND PROPERTY COMPILE_DEFINITIONS "BLOG_MIN_CC=${min_cc}")
  list(APPEND BLOG_TARGETS ${name})
  set(BLOG_TARGETS ${BLOG_TARGETS} PARENT_SCOPE)
endfunction()

add_blog_example(blog_01_tiled_copy                        01_tiled_copy.cu                          70)
add_blog_example(blog_02_sgemm_sm80                        02_sgemm_sm80.cu                          80)
add_blog_example(blog_03_wgmma_sm90                        03_wgmma_sm90.cu                          90)
add_blog_example(blog_04_wgmma_tma_sm90                    04_wgmma_tma_sm90.cu                      90)
add_blog_example(blog_05_blackwell_mma_sm100               05_blackwell_mma_sm100.cu                100)
add_blog_example(blog_06_blackwell_mma_tma_sm100           06_blackwell_mma_tma_sm100.cu            100)
add_blog_example(blog_07_blackwell_mma_tma_multicast_sm100 07_blackwell_mma_tma_multicast_sm100.cu  100)
add_blog_example(blog_08_blackwell_mma_tma_2sm_sm100       08_blackwell_mma_tma_2sm_sm100.cu        100)
add_blog_example(blog_09_blackwell_mma_tma_epi_sm100       09_blackwell_mma_tma_epi_sm100.cu        100)

# Headers from this directory are available to the Blackwell examples (example_utils.hpp).
foreach(target IN LISTS BLOG_TARGETS)
  target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

add_custom_target(blog_all DEPENDS ${BLOG_TARGETS})
