#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/codegen/local_accum.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/transforms/reorder.h"
#include "peachy_ir_agent/transforms/tile.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Local Accum for Naive Matmul") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  loop i: axis(0) in (0, 1024) 1:\n"
      "    loop j: axis(1) in (0, 1024) 1:\n"
      "      let C_i_j_0: float = 0;\n"
      "      loop k: axis(2) in (0, 512) 1:\n"
      "        C_i_j_0 = C_i_j_0 + A[i, k] * B[k, j];\n"
      "      C[i, j] = C_i_j_0;\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  CHECK_EQ(IrPrinter::print(LocalAccumTransform::apply(matmul_ir)),
           kExpectedStr);
}

TEST_CASE("Test Local Accum for Tiled Matmul") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 1\n"
      "  | axis(1) | = 1024, |T| = 1\n"
      "  | axis(2) | = 512, |T| = 1\n"
      "  define D0 = 4\n"
      "  define D1 = 4\n"
      "  define D2 = 4\n"
      "  loop i.T0: axis(0) in (0, 1024) D0:\n"
      "    loop i: axis(0) in (i.T0, i.T0 + D0) 1:\n"
      "      loop j.T0: axis(1) in (0, 1024) D1:\n"
      "        loop j: axis(1) in (j.T0, j.T0 + D1) 1:\n"
      "          let C_i_j_0: float = 0;\n"
      "          loop k.T0: axis(2) in (0, 512) D2:\n"
      "            loop k: axis(2) in (k.T0, k.T0 + D2) 1:\n"
      "              C_i_j_0 = C_i_j_0 + A[i, k] * B[k, j];\n"
      "          C[i, j] = C_i_j_0;\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");

  // Use these in our test. This is a bit hacky, but it will do.
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  LoopNodePtr jloop = std::static_pointer_cast<LoopNode>(iloop->body());
  LoopNodePtr kloop = std::static_pointer_cast<LoopNode>(jloop->body());

  TransformResult tile_iloop_result = TileTransform::apply(matmul_ir, *iloop);
  TransformResult tile_jloop_result =
      TileTransform::apply(tile_iloop_result.ir, *jloop);
  TransformResult tile_kloop_result =
      TileTransform::apply(tile_jloop_result.ir, *kloop);
  CHECK_EQ(IrPrinter::print(LocalAccumTransform::apply(tile_kloop_result.ir)),
           kExpectedStr);
}

TEST_CASE("Test No Local Accum") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  loop i: axis(0) in (0, 1024) 1:\n"
      "    loop k: axis(2) in (0, 512) 1:\n"
      "      loop j: axis(1) in (0, 1024) 1:\n"
      "        C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr jloop = std::static_pointer_cast<LoopNode>(
      std::static_pointer_cast<LoopNode>(matmul_ir->body())->body());
  TransformResult result = ReorderTransform::apply(matmul_ir, *jloop);
  CHECK_EQ(IrPrinter::print(LocalAccumTransform::apply(result.ir)),
           kExpectedStr);
}
}  // namespace peachyir
