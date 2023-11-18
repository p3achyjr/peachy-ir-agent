#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/transforms/reorder.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/transforms/tile.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Simple Reorder") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  loop j: axis(1) in (0, 1024) 1:\n"
      "    loop i: axis(0) in (0, 1024) 1:\n"
      "      loop k: axis(2) in (0, 512) 1:\n"
      "        C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  TransformResult result = ReorderTransform::apply(matmul_ir, *iloop);

  CHECK_MESSAGE(result, result.error_msg);
  CHECK_EQ(IrPrinter::print(result.ir), kExpectedStr);
}

TEST_CASE("Test Inner Reorder") {
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

  CHECK_MESSAGE(result, result.error_msg);
  CHECK_EQ(IrPrinter::print(result.ir), kExpectedStr);
}

TEST_CASE("Test Invalid Reorder") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 1\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  define D0 = 4\n"
      "  loop i.T0: axis(0) in (0, 1024) D0:\n"
      "    loop i: axis(0) in (i.T0, i.T0 + D0) 1:\n"
      "      loop j: axis(1) in (0, 1024) 1:\n"
      "        loop k: axis(2) in (0, 512) 1:\n"
      "          C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";

  // Tile once.
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  FunctionNodePtr ir =
      TileTransform::apply(
          matmul_ir, *std::static_pointer_cast<LoopNode>(matmul_ir->body()))
          .ir;

  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(ir->body());
  CHECK_FALSE(ReorderTransform::apply(matmul_ir, *iloop));
}

}  // namespace peachyir
