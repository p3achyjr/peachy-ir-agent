#include "peachy_ir_agent/kernels/matmul.h"

#include "peachy_ir_agent/ir.h"

namespace peachyir {
namespace kernels {

FunctionNode matmulIr(size_t m, size_t n, size_t k, std::string name) {
  // def matmul(C: (M, N), A: (M, K), B: (K, N)):
  //   loop "i" (0, M) 1 in
  //     loop "j" (0, N) 1 in
  //       loop "k" (0, K) 1 in
  //         C[i, j] = C[i, j] + A[i, k] * B[k, j]
  TensorVarNode c_var("C", {m, n});
  TensorVarNode a_var("A", {m, k});
  TensorVarNode b_var("B", {k, n});

  // Axes.
  InductionVarNode i_axis("i", 0);
  InductionVarNode j_axis("j", 1);
  InductionVarNode k_axis("k", 2);

  // Indices into tensors.
  IndexExpressionNode c_index({IndexExpressionNode::AxisIndex(i_axis),
                               IndexExpressionNode::AxisIndex(j_axis)});
  IndexExpressionNode a_index({IndexExpressionNode::AxisIndex(i_axis),
                               IndexExpressionNode::AxisIndex(k_axis)});
  IndexExpressionNode b_index({IndexExpressionNode::AxisIndex(k_axis),
                               IndexExpressionNode::AxisIndex(j_axis)});

  // RHS of accumulator.
  VarExprNode c_ref(TensorVarRefNode(c_var, c_index));
  VarExprNode a_ref(TensorVarRefNode(a_var, a_index));
  VarExprNode b_ref(TensorVarRefNode(b_var, b_index));
  BinopNode prod(a_ref, BinopNode::OpCode::kMul, b_ref);
  BinopNode accum(c_ref, BinopNode::OpCode::kAdd, prod);

  // LHS of accumulator.
  VarLocNode asgn_loc(TensorVarRefNode(c_var, c_index));

  // Accum.
  AsgnNode c_asgn(asgn_loc, accum);

  // `k` Loop.
  LoopNode k_loop(k_axis, 0, k, 1, c_asgn);

  // `j` Loop.
  LoopNode j_loop(j_axis, 0, n, 1, k_loop);

  // `i` Loop.
  LoopNode i_loop(i_axis, 0, m, 1, j_loop);

  // Function.
  return FunctionNode(
      name, {VarDeclNode(c_var), VarDeclNode(a_var), VarDeclNode(b_var)}, {},
      i_loop);
}

}  // namespace kernels
}  // namespace peachyir
