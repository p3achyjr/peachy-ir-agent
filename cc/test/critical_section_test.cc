#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/codegen/critical_section.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/transforms/parallel.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Non Parallel IR Returns Unchanged.") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  loop i: axis(0) in (0, 1024) 1:\n"
      "    loop j: axis(1) in (0, 1024) 1:\n"
      "      loop k: axis(2) in (0, 512) 1:\n"
      "        C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  CHECK_EQ(IrPrinter::print(CriticalSectionTransform::apply(matmul_ir)),
           kExpectedStr);
}

TEST_CASE("Test Parallel No Critical Section") {
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
  FunctionNodePtr parallel_ir = ParallelTransform::apply(matmul_ir, *iloop).ir;

  CHECK_EQ(IrPrinter::print(CriticalSectionTransform::apply(parallel_ir)),
           kExpectedStr);
}

TEST_CASE("Test Parallel Critical Section") {
  static const std::string kExpectedStr =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  | axis(1) | = 1024, |T| = 0\n"
      "  | axis(2) | = 512, |T| = 0\n"
      "  loop i: axis(0) in (0, 1024) 1:\n"
      "    loop j: axis(1) in (0, 1024) 1:\n"
      "      parallel k: axis(2) in (0, 512) 1:\n"
      "        critical:\n"
      "          C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  LoopNodePtr kloop = std::static_pointer_cast<LoopNode>(
      std::static_pointer_cast<LoopNode>(
          std::static_pointer_cast<LoopNode>(matmul_ir->body())->body())
          ->body());
  FunctionNodePtr parallel_ir = ParallelTransform::apply(matmul_ir, *kloop).ir;

  CHECK_EQ(IrPrinter::print(CriticalSectionTransform::apply(parallel_ir)),
           kExpectedStr);
}

TEST_CASE("Test Scalar Critical") {
  static const std::string kExpectedStr =
      "def f (x: float):\n"
      "  | axis(0) | = 1024, |T| = 0\n"
      "  parallel i: axis(0) in (0, 1024) 1:\n"
      "    critical:\n"
      "      x = i;\n";
  auto make_ir = []() {
    InductionVarNode i_var("i", 0);
    ScalarVarNode x_var("x", Type::kFloat);
    DefaultVarRefNode x_ref(x_var);
    AsgnNodePtr inner_write = AsgnNode::create(
        VarLocNode(x_ref), VarExprNode::create(DefaultVarRefNode(i_var)));
    LoopNodePtr loop_node =
        LoopNode::create(InductionVarNode("i", 0), 0, 1024, 1, inner_write,
                         true /* is_parallel */);
    return FunctionNode::create("f", {VarDeclNode(x_var)}, {}, {{1024, 0}},
                                loop_node);
  };

  FunctionNodePtr ir = make_ir();
  FunctionNodePtr parallel_ir =
      ParallelTransform::apply(
          ir, *(std::static_pointer_cast<LoopNode>(ir->body())))
          .ir;
  CHECK_EQ(IrPrinter::print(CriticalSectionTransform::apply(parallel_ir)),
           kExpectedStr);
}
}  // namespace peachyir
