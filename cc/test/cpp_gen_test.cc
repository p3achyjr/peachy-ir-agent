#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/codegen/cpp_gen.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/codegen/critical_section.h"
#include "peachy_ir_agent/codegen/local_accum.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/transforms/parallel.h"
#include "peachy_ir_agent/transforms/reorder.h"
#include "peachy_ir_agent/transforms/tile.h"
#include "peachy_ir_agent/transforms/tile_resize.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Naive Matmul.") {
  static const std::string kExpectedStr =
      "void matmul (float* __restrict C /* (1024, 1024) */, float* __restrict "
      "A /* "
      "(1024, 512) */, float* __restrict B /* (512, 1024) */) {\n"
      "  for (size_t i = 0; i < 1024; i += 1) {\n"
      "    for (size_t j = 0; j < 1024; j += 1) {\n"
      "      for (size_t k = 0; k < 512; k += 1) {\n"
      "        C[(1024 * i) + j] = C[(1024 * i) + j] + A[(512 * i) + k] * "
      "B[(1024 * k) + j];\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  CHECK_EQ(kExpectedStr, CppGen::apply(matmul_ir));
}

TEST_CASE("Test Parallel Tiled Critical LocalAccum Matmul.") {
  static const std::string kExpectedStr =
      R"(void matmul (float* __restrict C /* (1024, 1024) */, float* __restrict A /* (1024, 512) */, float* __restrict B /* (512, 1024) */) {
  static constexpr size_t D0 = 4;
  static constexpr size_t D1 = 4;
  static constexpr size_t D2 = 4;
  for (size_t i_T0 = 0; i_T0 < 1024; i_T0 += D0) {
#pragma omp parallel for
    for (size_t j_T0 = 0; j_T0 < 1024; j_T0 += D1) {
      for (size_t k_T0 = 0; k_T0 < 512; k_T0 += D2) {
        for (size_t i = i_T0; i < i_T0 + D0; i += 1) {
          for (size_t j = j_T0; j < j_T0 + D1; j += 1) {
            float C_i_j_0 = 0;
            for (size_t k = k_T0; k < k_T0 + D2; k += 1) {
              C_i_j_0 = C_i_j_0 + A[(512 * i) + k] * B[(1024 * k) + j];
            }
            C[(1024 * i) + j] = C_i_j_0;
          }
        }
      }
    }
  }
}
)";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");

  // Hacky, but use these in our test.
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  LoopNodePtr jloop = std::static_pointer_cast<LoopNode>(iloop->body());
  LoopNodePtr kloop = std::static_pointer_cast<LoopNode>(jloop->body());

  TransformResult tile_iloop_result = TileTransform::apply(matmul_ir, *iloop);
  TransformResult tile_jloop_result =
      TileTransform::apply(tile_iloop_result.ir, *jloop);
  TransformResult parallel_ij_result = ParallelTransform::apply(
      tile_jloop_result.ir,
      *(std::static_pointer_cast<LoopNode>(
            std::static_pointer_cast<LoopNode>(tile_jloop_result.ir->body())
                ->body())
            ->body()));
  TransformResult tile_kloop_result =
      TileTransform::apply(parallel_ij_result.ir, *kloop);
  TransformResult reorder_ij_result =
      ReorderTransform::apply(tile_kloop_result.ir, *iloop);
  TransformResult reorder_jk_result =
      ReorderTransform::apply(reorder_ij_result.ir, *jloop);
  TransformResult reorder_ik_result =
      ReorderTransform::apply(reorder_jk_result.ir, *iloop);
  FunctionNodePtr ir = LocalAccumTransform::apply(reorder_ik_result.ir);
  ir = CriticalSectionTransform::apply(ir);
  CHECK_EQ(kExpectedStr, CppGen::apply(ir));
}
}  // namespace peachyir
