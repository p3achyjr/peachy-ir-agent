#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/transforms/tile_resize.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/transforms/tile.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Simple Tile Resize") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 1\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  define D0 = 8\n"
      "  loop i.T0: axis(0) in (0, 1024) D0:\n"
      "    loop i: axis(0) in (i.T0, i.T0 + D0) 1:\n"
      "      loop j: axis(1) in (0, 1024) 1:\n"
      "        loop k: axis(2) in (0, 512) 1:\n"
      "          C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  TransformResult result = TileTransform::apply(matmul_ir, *iloop);

  CHECK_MESSAGE(result, result.error_msg);
  iloop = std::static_pointer_cast<LoopNode>(result.ir->body());
  result = TileResizeTransform::apply(result.ir, *iloop);
  CHECK_EQ(IrPrinter::print(result.ir), kExpectedStr);
}

TEST_CASE("Test Repeated Tile Resize") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 1\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  define D0 = 16\n"
      "  loop i.T0: axis(0) in (0, 1024) D0:\n"
      "    loop i: axis(0) in (i.T0, i.T0 + D0) 1:\n"
      "      loop j: axis(1) in (0, 1024) 1:\n"
      "        loop k: axis(2) in (0, 512) 1:\n"
      "          C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  TransformResult result = TileTransform::apply(matmul_ir, *iloop);

  CHECK_MESSAGE(result, result.error_msg);
  iloop = std::static_pointer_cast<LoopNode>(result.ir->body());
  result = TileResizeTransform::apply(result.ir, *iloop);
  CHECK_MESSAGE(result, result.error_msg);
  iloop = std::static_pointer_cast<LoopNode>(result.ir->body());
  result = TileResizeTransform::apply(result.ir, *iloop);
  CHECK_MESSAGE(result, result.error_msg);
  CHECK_EQ(IrPrinter::print(result.ir), kExpectedStr);
}

TEST_CASE("Test Multi-Axis Tile Resize") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 1\n"
      "  | axis(1) | = 1024, |T| = 1\n"
      "  | axis(2) | = 512, |T| = 1\n"
      "  define D0 = 8\n"
      "  define D1 = 8\n"
      "  define D2 = 8\n"
      "  loop i.T0: axis(0) in (0, 1024) D0:\n"
      "    loop i: axis(0) in (i.T0, i.T0 + D0) 1:\n"
      "      loop j.T0: axis(1) in (0, 1024) D1:\n"
      "        loop j: axis(1) in (j.T0, j.T0 + D1) 1:\n"
      "          loop k.T0: axis(2) in (0, 512) D2:\n"
      "            loop k: axis(2) in (k.T0, k.T0 + D2) 1:\n"
      "              C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");

  // Use these in our test. This is a bit hacky, but it will do.
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  LoopNodePtr jloop = std::static_pointer_cast<LoopNode>(iloop->body());
  LoopNodePtr kloop = std::static_pointer_cast<LoopNode>(jloop->body());

  TransformResult result = TileTransform::apply(matmul_ir, *iloop);
  CHECK_MESSAGE(result, result.error_msg);

  result = TileTransform::apply(result.ir, *jloop);
  CHECK_MESSAGE(result, result.error_msg);

  result = TileTransform::apply(result.ir, *kloop);
  CHECK_MESSAGE(result, result.error_msg);

  LoopNodePtr tiled_iloop =
      std::static_pointer_cast<LoopNode>(result.ir->body());
  LoopNodePtr tiled_jloop = std::static_pointer_cast<LoopNode>(
      std::static_pointer_cast<LoopNode>(tiled_iloop->body())->body());
  LoopNodePtr tiled_kloop = std::static_pointer_cast<LoopNode>(
      std::static_pointer_cast<LoopNode>(tiled_jloop->body())->body());

  result = TileResizeTransform::apply(result.ir, *tiled_iloop);
  CHECK_MESSAGE(result, result.error_msg);

  result = TileResizeTransform::apply(result.ir, *tiled_jloop);
  CHECK_MESSAGE(result, result.error_msg);

  result = TileResizeTransform::apply(result.ir, *tiled_kloop);
  CHECK_MESSAGE(result, result.error_msg);
  CHECK_EQ(IrPrinter::print(result.ir), kExpectedStr);
}

TEST_CASE("Test Invalid Tile Resize") {
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());

  TransformResult result = TileResizeTransform::apply(matmul_ir, *iloop);
  CHECK_FALSE(TileResizeTransform::apply(result.ir, *iloop));
}

TEST_CASE("Test Tile Resize Non-Divides") {
  // Tile first. 1020 % 8 != 0
  FunctionNodePtr matmul_ir = kernels::matmulIr(1020, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  TransformResult result = TileTransform::apply(matmul_ir, *iloop);

  CHECK_MESSAGE(result, result.error_msg);
  iloop = std::static_pointer_cast<LoopNode>(result.ir->body());
  CHECK_FALSE(TileResizeTransform::apply(result.ir, *iloop));
}
}  // namespace peachyir
