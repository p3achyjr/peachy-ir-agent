#include "peachy_ir_agent/kernels/matmul.h"

#include <iostream>

#include "peachy_ir_agent/ir.h"

namespace peachyir {
namespace kernels {

FunctionNodePtr matmulIr(size_t m, size_t n, size_t k, std::string name) {
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
  VarExprNodePtr c_ref = VarExprNode::create(TensorVarRefNode(c_var, c_index));
  VarExprNodePtr a_ref = VarExprNode::create(TensorVarRefNode(a_var, a_index));
  VarExprNodePtr b_ref = VarExprNode::create(TensorVarRefNode(b_var, b_index));
  BinopNodePtr prod = BinopNode::create(a_ref, BinopNode::OpCode::kMul, b_ref);
  BinopNodePtr accum = BinopNode::create(c_ref, BinopNode::OpCode::kAdd, prod);

  // LHS of accumulator.
  VarLocNode asgn_loc(TensorVarRefNode(c_var, c_index));

  // Accum.
  AsgnNodePtr c_asgn = AsgnNode::create(asgn_loc, accum);

  // `k` Loop.
  LoopNodePtr k_loop = LoopNode::create(k_axis, 0, k, 1, c_asgn);

  // `j` Loop.
  LoopNodePtr j_loop = LoopNode::create(j_axis, 0, n, 1, k_loop);

  // `i` Loop.
  LoopNodePtr i_loop = LoopNode::create(i_axis, 0, m, 1, j_loop);

  // Function.
  return FunctionNode::create(
      name,
      {VarDeclNode(c_var, true), VarDeclNode(a_var, false),
       VarDeclNode(b_var, false)},
      {}, {{m, 0}, {n, 0}, {k, 0}}, i_loop, false /* is_parallel */);
}

}  // namespace kernels
}  // namespace peachyir
