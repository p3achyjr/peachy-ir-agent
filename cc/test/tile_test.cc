#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/transforms/tile.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Simple Tile") {
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
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());

  Result can_tile = TileTransform::canApply(matmul_ir, *iloop);
  CHECK_MESSAGE(can_tile, can_tile.error_msg);

  FunctionNodePtr tiled_matmul_ir = TileTransform::apply(matmul_ir, *iloop);
  CHECK_EQ(IrPrinter::print(tiled_matmul_ir), kExpectedStr);
}

TEST_CASE("Test Multi-Level Tile") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 2\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  define D0 = 4\n"
      "  define D1 = 8\n"
      "  loop i.T1: axis(0) in (0, 1024) D1:\n"
      "    loop i.T0: axis(0) in (i.T1, i.T1 + D1) D0:\n"
      "      loop i: axis(0) in (i.T0, i.T0 + D0) 1:\n"
      "        loop j: axis(1) in (0, 1024) 1:\n"
      "          loop k: axis(2) in (0, 512) 1:\n"
      "            C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());

  Result can_tile = TileTransform::canApply(matmul_ir, *iloop);
  CHECK_MESSAGE(can_tile, can_tile.error_msg);

  FunctionNodePtr tiled_ir = TileTransform::apply(matmul_ir, *iloop);
  LoopNodePtr iloop_t0 = std::static_pointer_cast<LoopNode>(tiled_ir->body());

  Result can_multi_level_tile = TileTransform::canApply(tiled_ir, *iloop_t0);
  CHECK_MESSAGE(can_multi_level_tile, can_multi_level_tile.error_msg);

  FunctionNodePtr multi_level_tiled_ir =
      TileTransform::apply(tiled_ir, *iloop_t0);
  CHECK_EQ(IrPrinter::print(multi_level_tiled_ir), kExpectedStr);
}
}  // namespace peachyir
