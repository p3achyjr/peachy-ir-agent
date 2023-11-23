#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/transforms/parallel.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Parallelize iloop") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  parallel i: axis(0) in (0, 1024) 1:\n"
      "    loop j: axis(1) in (0, 1024) 1:\n"
      "      loop k: axis(2) in (0, 512) 1:\n"
      "        C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  TransformResult result = ParallelTransform::apply(matmul_ir, *iloop);

  CHECK_EQ(IrPrinter::print(result.ir), kExpectedStr);
}

TEST_CASE("Test Parallelize jloop") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  loop i: axis(0) in (0, 1024) 1:\n"
      "    parallel j: axis(1) in (0, 1024) 1:\n"
      "      loop k: axis(2) in (0, 512) 1:\n"
      "        C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr jloop = std::static_pointer_cast<LoopNode>(
      std::static_pointer_cast<LoopNode>(matmul_ir->body())->body());
  TransformResult result = ParallelTransform::apply(matmul_ir, *jloop);

  CHECK_EQ(IrPrinter::print(result.ir), kExpectedStr);
}

TEST_CASE("Test Cannot Repeatedly Parallelize") {
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr iloop = std::static_pointer_cast<LoopNode>(matmul_ir->body());
  TransformResult result = ParallelTransform::apply(matmul_ir, *iloop);

  CHECK_FALSE(ParallelTransform::apply(result.ir, *iloop));
  CHECK_FALSE(ParallelTransform::apply(
      result.ir, *std::static_pointer_cast<LoopNode>(iloop->body())));

  // Check persistence.
  CHECK(ParallelTransform::apply(matmul_ir, *iloop));
}
}  // namespace peachyir
