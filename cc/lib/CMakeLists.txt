set(
  SOURCES
  ir.cc
  rand.cc
  codegen/cpp_gen.cc
  codegen/critical_section.cc
  codegen/local_accum.cc
  codegen/codegen.cc
  kernels/matmul.cc
  search/cpu_cost_model.cc
  search/random_search.cc
  transforms/enclosing_loop_stride_finder.cc
  transforms/parallel.cc
  transforms/reorder.cc
  transforms/tile_common.cc
  transforms/tile_resize.cc
  transforms/tile.cc
  transforms/reorder.cc
  transforms/util.cc
)

add_library(peachy_ir_agent ${SOURCES})
target_include_directories(peachy_ir_agent PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(peachy_ir_agent PRIVATE ${CMAKE_BINARY_DIR})
