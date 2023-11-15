#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/ir.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/visitor/ir_printer.h"
#include "peachy_ir_agent/visitor/path_copy_visitor.h"

namespace peachyir {
namespace {
static const std::string kExpectedMatmulStr =
    "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
    "  | axis(0) | = 1024, |T| = 0\n"
    "  | axis(1) | = 1024, |T| = 0\n"
    "  | axis(2) | = 512, |T| = 0\n"
    "  loop i: axis(0) in (0, 1024) 1:\n"
    "    loop j: axis(1) in (0, 1024) 1:\n"
    "      loop k: axis(2) in (0, 512) 1:\n"
    "        C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";

/*
 * Visitor that sets `changed` to true, thereby forcing a path to be copied.
 */
class ForceCopyVisitor : public PathCopyVisitor<ForceCopyVisitor> {
 public:
  UniquePtrResult<TensorVarRefNode> completeTensorVarRef(
      const TensorVarRefNode& node, UniquePtrResult<TensorVarNode>&& var_result,
      Result<IndexExpressionNode> index_expr_result) {
    return UniquePtrResult<TensorVarRefNode>(
        true, std::make_unique<TensorVarRefNode>(*var_result.node,
                                                 index_expr_result.node));
  }

  static FunctionNodePtr copy(FunctionNodePtr node) {
    ForceCopyVisitor visitor;
    return extract(visitor.visit(node));
  }
};
}  // namespace

TEST_CASE("Test Matmul") {
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  std::string matmul_str = IrPrinter::print(matmul_ir);

  CHECK_EQ(kExpectedMatmulStr, matmul_str);
}

TEST_CASE("Test Path Copy") {
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  FunctionNodePtr matmul_ir_trivial_copy =
      TrivialPathCopyVisitor::copy(matmul_ir);

  CHECK_EQ(matmul_ir.get(), matmul_ir_trivial_copy.get());
  CHECK_EQ(IrPrinter::print(matmul_ir),
           kExpectedMatmulStr);  // check original is unchanged.
  CHECK_EQ(IrPrinter::print(matmul_ir_trivial_copy),
           kExpectedMatmulStr);  // check new is the same.

  FunctionNodePtr matmul_ir_path_copy = ForceCopyVisitor::copy(matmul_ir);

  CHECK_NE(matmul_ir.get(),
           matmul_ir_path_copy.get());  // check pointers are different.
  CHECK_EQ(IrPrinter::print(matmul_ir),
           kExpectedMatmulStr);  // check original is unchanged.
  CHECK_EQ(IrPrinter::print(matmul_ir_path_copy),
           kExpectedMatmulStr);  // check new is the same.
}
}  // namespace peachyir
