#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/transforms/tile.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/transforms/parallel.h"
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
  TransformResult tile_result = TileTransform::apply(matmul_ir, *iloop);

  CHECK_MESSAGE(tile_result, tile_result.error_msg);
  CHECK_EQ(IrPrinter::print(tile_result.ir), kExpectedStr);
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

  TransformResult tile_result = TileTransform::apply(matmul_ir, *iloop);
  CHECK_MESSAGE(tile_result, tile_result.error_msg);

  FunctionNodePtr tiled_ir = tile_result.ir;
  LoopNodePtr iloop_t0 = std::static_pointer_cast<LoopNode>(tiled_ir->body());

  TransformResult multi_level_tiled_result =
      TileTransform::apply(tiled_ir, *iloop_t0);
  CHECK_MESSAGE(multi_level_tiled_result, multi_level_tiled_result.error_msg);

  FunctionNodePtr multi_level_tiled_ir = multi_level_tiled_result.ir;
  CHECK_EQ(IrPrinter::print(multi_level_tiled_ir), kExpectedStr);
}

TEST_CASE("Test Multi-Axis Tile") {
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
      "          loop k.T0: axis(2) in (0, 512) D2:\n"
      "            loop k: axis(2) in (k.T0, k.T0 + D2) 1:\n"
      "              C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");

  // Use these in our test. This is a bit hacky, but it will do.
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  LoopNodePtr jloop = std::static_pointer_cast<LoopNode>(iloop->body());
  LoopNodePtr kloop = std::static_pointer_cast<LoopNode>(jloop->body());

  TransformResult tile_iloop_result = TileTransform::apply(matmul_ir, *iloop);
  CHECK_MESSAGE(tile_iloop_result, tile_iloop_result.error_msg);

  TransformResult tile_jloop_result =
      TileTransform::apply(tile_iloop_result.ir, *jloop);
  CHECK_MESSAGE(tile_jloop_result, tile_jloop_result.error_msg);

  TransformResult tile_kloop_result =
      TileTransform::apply(tile_jloop_result.ir, *kloop);
  CHECK_MESSAGE(tile_kloop_result, tile_kloop_result.error_msg);
  CHECK_EQ(IrPrinter::print(tile_kloop_result.ir), kExpectedStr);
}

TEST_CASE("Test Tile Preserves Parallelism") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 2\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  define D0 = 4\n"
      "  define D1 = 8\n"
      "  loop i.T1: axis(0) in (0, 1024) D1:\n"
      "    parallel i.T0: axis(0) in (i.T1, i.T1 + D1) D0:\n"
      "      loop i: axis(0) in (i.T0, i.T0 + D0) 1:\n"
      "        loop j: axis(1) in (0, 1024) 1:\n"
      "          loop k: axis(2) in (0, 512) 1:\n"
      "            C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());

  TransformResult tile_result = TileTransform::apply(matmul_ir, *iloop);
  CHECK_MESSAGE(tile_result, tile_result.error_msg);

  FunctionNodePtr tiled_ir = tile_result.ir;
  FunctionNodePtr ptiled_ir =
      ParallelTransform::apply(tiled_ir, *(tiled_ir->body())).ir;
  LoopNodePtr iloop_t0 = std::static_pointer_cast<LoopNode>(ptiled_ir->body());

  TransformResult multi_level_tiled_result =
      TileTransform::apply(ptiled_ir, *iloop_t0);
  CHECK_MESSAGE(multi_level_tiled_result, multi_level_tiled_result.error_msg);

  FunctionNodePtr multi_level_tiled_ir = multi_level_tiled_result.ir;
  CHECK_EQ(IrPrinter::print(multi_level_tiled_ir), kExpectedStr);
}

TEST_CASE("Test Invalid Tile") {
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());

  TransformResult tile_result = TileTransform::apply(matmul_ir, *iloop);
  CHECK_FALSE(TileTransform::apply(tile_result.ir, *iloop));
}
}  // namespace peachyir
