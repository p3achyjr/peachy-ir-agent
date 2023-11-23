set(
  SOURCES
  ir.cc
  codegen/cpp_gen.cc
  codegen/critical_section.cc
  codegen/local_accum.cc
  kernels/matmul.cc
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
