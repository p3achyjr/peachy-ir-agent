#pragma once

#include <algorithm>
#include <functional>
#include <numeric>

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir.h"

#define DELEGATE(METHOD, CLASS) \
  return visitor()->visit##METHOD(static_cast<const CLASS&>(node))

#define RETURN_ORIGINAL_IF_UNCHANGED(ResultT, node, ...) \
  do {                                                   \
    if (!changed(__VA_ARGS__)) {                         \
      return ResultT{false, node};                       \
    }                                                    \
  } while (0);

namespace peachyir {

/*
 * Visitor that path-copies an IR.
 *
 * Traverses IR, and returns a tuple `changed, std::shared_ptr<IrNode>`. If
 * `changed` is true, then we have created a new node. We propagate this state
 * information all the way to the top node.
 *
 * The default behavior will traverse the entire IR, but return the IR
 * unchanged. This is because in the default behavior, there are no writes.
 */
template <typename VisitorT>
class PathCopyVisitor {
 public:
  template <typename NodeT>
  struct PtrResult {
    bool changed;
    std::shared_ptr<NodeT> node;

    PtrResult(bool changed, std::shared_ptr<NodeT> node)
        : changed(changed), node(node) {}

    template <typename DerivedNodeT,
              std::enable_if_t<
                  std::is_convertible_v<std::decay_t<DerivedNodeT>*, NodeT*>,
                  bool> = true>
    PtrResult(const PtrResult<DerivedNodeT>& other)
        : changed(other.changed), node(other.node) {}
  };

  template <typename NodeT>
  struct UniquePtrResult {
    bool changed;
    std::unique_ptr<NodeT> node;

    UniquePtrResult(bool changed, std::unique_ptr<NodeT> node)
        : changed(changed), node(std::move(node)) {}

    template <typename DerivedNodeT,
              std::enable_if_t<
                  std::is_convertible_v<std::decay_t<DerivedNodeT>*, NodeT*>,
                  bool> = true>
    UniquePtrResult(const UniquePtrResult<DerivedNodeT>& other)
        : changed(other.changed), node(other.node->clone()) {}
  };

  template <typename NodeT>
  struct Result {
    bool changed;
    NodeT node;

    Result(bool changed, NodeT node) : changed(changed), node(node) {}
  };

  // Generic sequence visit methods.
  template <typename NodeT>
  std::vector<PtrResult<NodeT>> visit(
      const std::vector<std::shared_ptr<NodeT>>& nodes) {
    std::vector<PtrResult<NodeT>> node_results;
    for (std::shared_ptr<NodeT> node : nodes) {
      node_results.emplace_back(visitor()->visit(node));
    }

    return node_results;
  }

  template <typename NodeT>
  std::vector<Result<NodeT>> visit(const std::vector<NodeT>& nodes) {
    std::vector<Result<NodeT>> node_results;
    for (const NodeT& node : nodes) {
      node_results.emplace_back(visitor()->visit(node));
    }

    return node_results;
  }

  // Pointer visit methods for final types.
  PtrResult<FunctionNode> visit(std::shared_ptr<FunctionNode> node) {
    visitor()->visitFunction(*node);
    std::vector<Result<VarDeclNode>> arg_results =
        visitor()->visit(node->args());
    std::vector<Result<DefineNode>> define_results =
        visitor()->visit(node->defines());
    PtrResult<StmtNode> body_result = visitor()->visit(node->body());
    return visitor()->completeFunction(node, arg_results, define_results,
                                       body_result);
  }

  PtrResult<BinopNode> visit(std::shared_ptr<BinopNode> node) {
    visitor()->visitBinop(*node);
    PtrResult<ExprNode> lhs_result = visitor()->visit(node->lhs());
    PtrResult<ExprNode> rhs_result = visitor()->visit(node->rhs());
    return visitor()->completeBinop(node, lhs_result, rhs_result);
  }

  PtrResult<UnopNode> visit(std::shared_ptr<UnopNode> node) {
    visitor()->visitUnop(*node);
    PtrResult<ExprNode> expr_result = visitor()->visit(node->expr());
    return visitor()->completeUnop(node, expr_result);
  }

  PtrResult<VarExprNode> visit(std::shared_ptr<VarExprNode> node) {
    visitor()->visitVarExpr(*node);
    return visitor()->completeVarExpr(node, visitor()->visit(node->var_ref()));
  }

  PtrResult<SeqNode> visit(std::shared_ptr<SeqNode> node) {
    visitor()->visitSeq(*node);
    std::vector<PtrResult<StmtNode>> stmt_results;
    return visitor()->completeSeq(node, stmt_results);
  }

  PtrResult<NopNode> visit(std::shared_ptr<NopNode> node) {
    visitor()->visitNop(*node);
    return PtrResult<NopNode>(false, node);
  }

  PtrResult<LetNode> visit(std::shared_ptr<LetNode> node) {
    visitor()->visitLet(*node);
    Result<VarDeclNode> var_decl_result = visitor()->visit(node->var_decl());
    PtrResult<ExprNode> expr_result = visitor()->visit(node->expr());
    PtrResult<StmtNode> scope_result = visitor()->visit(node->scope());
    return visitor()->completeLet(node, var_decl_result, expr_result,
                                  scope_result);
  }

  PtrResult<AsgnNode> visit(std::shared_ptr<AsgnNode> node) {
    visitor()->visitAsgn(*node);
    Result<VarLocNode> var_loc_result = visitor()->visit(node->var_loc());
    PtrResult<ExprNode> expr_result = visitor()->visit(node->expr());
    return visitor()->completeAsgn(node, var_loc_result, expr_result);
  }

  PtrResult<LoopNode> visit(std::shared_ptr<LoopNode> node) {
    visitor()->visitLoop(*node);
    UniquePtrResult<InductionVarNode> induction_var_result =
        visitor()->visit(node->induction_var());
    PtrResult<StmtNode> body_result = visitor()->visit(node->body());
    return visitor()->completeLoop(node, std::move(induction_var_result),
                                   body_result);
  }

  PtrResult<LoopNode> visit(std::shared_ptr<ParLoopNode> node) {
    visitor()->visitParLoop(*node);
    UniquePtrResult<InductionVarNode> induction_var_result =
        visitor()->visit(node->induction_var());
    PtrResult<StmtNode> body_result = visitor()->visit(node->body());
    return visitor()->completeParLoop(node, std::move(induction_var_result),
                                      body_result);
  }

  PtrResult<CriticalSectionNode> visit(
      std::shared_ptr<CriticalSectionNode> node) {
    visitor()->visitCriticalSection(*node);
    std::vector<PtrResult<AsgnNode>> critical_write_results =
        visitor()->visit(node->critical_writes());
    return visitor()->completeCriticalSection(node, critical_write_results);
  }

  // Pointer visit methods for subclassed types. Contains dispatch logic.
  PtrResult<ExprNode> visit(std::shared_ptr<ExprNode> node) {
    switch (node->kind()) {
      case IrNode::Kind::kBinop:
        return visitor()->visit(std::static_pointer_cast<BinopNode>(node));
      case IrNode::Kind::kUnop:
        return visitor()->visit(std::static_pointer_cast<UnopNode>(node));
      case IrNode::Kind::kVarExpr:
        return visitor()->visit(std::static_pointer_cast<VarExprNode>(node));
      default:
        ABORT("Invalid ExprNode. OpCode: `%s`", str(node->kind()).c_str());
    }
  }

  PtrResult<StmtNode> visit(std::shared_ptr<StmtNode> node) {
    switch (node->kind()) {
      case IrNode::Kind::kSeq:
        return visitor()->visit(std::static_pointer_cast<SeqNode>(node));
      case IrNode::Kind::kNop:
        return visitor()->visit(std::static_pointer_cast<NopNode>(node));
      case IrNode::Kind::kLet:
        return visitor()->visit(std::static_pointer_cast<LetNode>(node));
      case IrNode::Kind::kAsgn:
        return visitor()->visit(std::static_pointer_cast<AsgnNode>(node));
      case IrNode::Kind::kLoop:
        return visitor()->visit(std::static_pointer_cast<LoopNode>(node));
      case IrNode::Kind::kParLoop:
        return visitor()->visit(std::static_pointer_cast<ParLoopNode>(node));
      default:
        ABORT("Invalid StmtNode. OpCode: `%s`", str(node->kind()).c_str());
    }
  }

  // Value visit methods.
  Result<DefineNode> visit(const DefineNode& node) {
    visitor()->visitDefine(node);
    return visitor()->completeDefine(node);
  }

  UniquePtrResult<ScalarVarNode> visit(const ScalarVarNode& node) {
    visitor()->visitScalarVar(node);
    return visitor()->completeScalarVar(node);
  }

  UniquePtrResult<InductionVarNode> visit(const InductionVarNode& node) {
    visitor()->visitInductionVar(node);
    return visitor()->completeInductionVar(node);
  }

  UniquePtrResult<TensorVarNode> visit(const TensorVarNode& node) {
    visitor()->visitTensorVar(node);
    return visitor()->completeTensorVar(node);
  }

  Result<VarDeclNode> visit(const VarDeclNode& node) {
    visitor()->visitVarDecl(node);
    return visitor()->completeVarDecl(node, visitor()->visit(node.var()));
  }

  Result<VarLocNode> visit(const VarLocNode& node) {
    visitor()->visitVarLoc(node);
    return visitor()->completeVarLoc(node, visitor()->visit(node.var_ref()));
  }

  UniquePtrResult<DefaultVarRefNode> visit(const DefaultVarRefNode& node) {
    visitor()->visitDefaultVarRef(node);
    return visitor()->completeDefaultVarRef(node, visitor()->visit(node.var()));
  }

  UniquePtrResult<TensorVarRefNode> visit(const TensorVarRefNode& node) {
    visitor()->visitTensorVarRef(node);
    UniquePtrResult<TensorVarNode> var_result = visitor()->visit(node.var());
    Result<IndexExpressionNode> index_expr_result =
        visitor()->visit(node.index_expr());
    return visitor()->completeTensorVarRef(node, std::move(var_result),
                                           index_expr_result);
  }

  Result<IndexExpressionNode> visit(const IndexExpressionNode& node) {
    visitor()->visitIndexExpression(node);
    return visitor()->completeIndexExpression(node);
  }

  // Value visit methods for subclassed types. Contains dispatch logic.
  UniquePtrResult<VarNode> visit(const VarNode& node) {
    switch (node.kind()) {
      case IrNode::Kind::kScalarVar:
        return visitor()->visit(static_cast<const ScalarVarNode&>(node));
      case IrNode::Kind::kInductionVar:
        return visitor()->visit(static_cast<const InductionVarNode&>(node));
      case IrNode::Kind::kTensorVar:
        return visitor()->visit(static_cast<const TensorVarNode&>(node));
      default:
        ABORT("Invalid VarNode. OpCode: `%s`", str(node.kind()).c_str());
    }
  }

  UniquePtrResult<VarRefNode> visit(const VarRefNode& node) {
    switch (node.kind()) {
      case IrNode::Kind::kDefaultVarRef:
        return visitor()->visit(static_cast<const DefaultVarRefNode&>(node));
      case IrNode::Kind::kTensorVarRef:
        return visitor()->visit(static_cast<const TensorVarRefNode&>(node));
      default:
        ABORT("Invalid VarRefNode. OpCode: `%s`", str(node.kind()).c_str());
    }
  }

  // Pre-order callbacks.
  void visitFunction(const FunctionNode& node) { DELEGATE(IrNode, IrNode); }
  void visitDefine(const DefineNode& node) { DELEGATE(IrNode, IrNode); }
  void visitVar(const VarNode& node) { DELEGATE(IrNode, IrNode); }
  void visitScalarVar(const ScalarVarNode& node) { DELEGATE(Var, VarNode); }
  void visitInductionVar(const InductionVarNode& node) {
    DELEGATE(Var, VarNode);
  }
  void visitTensorVar(const TensorVarNode& node) { DELEGATE(Var, VarNode); }
  void visitVarDecl(const VarDeclNode& node) { DELEGATE(IrNode, IrNode); }
  void visitVarLoc(const VarLocNode& node) { DELEGATE(IrNode, IrNode); }
  void visitVarRef(const VarRefNode& node) { DELEGATE(IrNode, IrNode); }
  void visitDefaultVarRef(const DefaultVarRefNode& node) {
    DELEGATE(VarRef, VarRefNode);
  }
  void visitTensorVarRef(const TensorVarRefNode& node) {
    DELEGATE(VarRef, VarRefNode);
  }
  void visitIndexExpression(const IndexExpressionNode& node) {
    DELEGATE(IrNode, IrNode);
  }
  void visitExpr(const ExprNode& node) { DELEGATE(IrNode, IrNode); }
  void visitBinop(const BinopNode& node) { DELEGATE(Expr, ExprNode); }
  void visitUnop(const UnopNode& node) { DELEGATE(Expr, ExprNode); }
  void visitVarExpr(const VarExprNode& node) { DELEGATE(Expr, ExprNode); }
  void visitStmt(const StmtNode& node) { DELEGATE(IrNode, IrNode); }
  void visitSeq(const SeqNode& node) { DELEGATE(Stmt, StmtNode); }
  void visitNop(const NopNode& node) { DELEGATE(Stmt, StmtNode); }
  void visitLet(const LetNode& node) { DELEGATE(Stmt, StmtNode); }
  void visitAsgn(const AsgnNode& node) { DELEGATE(Stmt, StmtNode); }
  void visitLoop(const LoopNode& node) { DELEGATE(Stmt, StmtNode); }
  void visitParLoop(const ParLoopNode& node) { DELEGATE(Stmt, StmtNode); }
  void visitCriticalSection(const CriticalSectionNode& node) {
    DELEGATE(Stmt, StmtNode);
  }
  void visitIrNode(const IrNode& node) {}

  // Post-order callbacks.
  PtrResult<FunctionNode> completeFunction(
      std::shared_ptr<FunctionNode> node,
      std::vector<Result<VarDeclNode>> arg_results,
      std::vector<Result<DefineNode>> define_results,
      PtrResult<StmtNode> body_result) {
    RETURN_ORIGINAL_IF_UNCHANGED(PtrResult<FunctionNode>, node, arg_results,
                                 define_results, body_result);
    return PtrResult<FunctionNode>(
        true, FunctionNode::create(node->name(), extract(arg_results),
                                   extract(define_results), node->axes_info(),
                                   extract(body_result)));
  }

  PtrResult<BinopNode> completeBinop(std::shared_ptr<BinopNode> node,
                                     PtrResult<ExprNode> lhs_result,
                                     PtrResult<ExprNode> rhs_result) {
    RETURN_ORIGINAL_IF_UNCHANGED(PtrResult<BinopNode>, node, lhs_result,
                                 rhs_result);
    return PtrResult<BinopNode>(
        true, BinopNode::create(extract(lhs_result), node->op(),
                                extract(rhs_result)));
  }

  PtrResult<UnopNode> completeUnop(std::shared_ptr<UnopNode> node,
                                   PtrResult<ExprNode> expr_result) {
    RETURN_ORIGINAL_IF_UNCHANGED(PtrResult<UnopNode>, node, expr_result);
    return PtrResult<UnopNode>(
        true, UnopNode::create(node->op(), extract(expr_result)));
  }

  PtrResult<VarExprNode> completeVarExpr(
      std::shared_ptr<VarExprNode> node,
      UniquePtrResult<VarRefNode>&& var_ref_result) {
    if (!var_ref_result.changed) {
      return PtrResult<VarExprNode>(false, node);
    }

    return PtrResult<VarExprNode>(true,
                                  VarExprNode::create(*(var_ref_result.node)));
  }

  PtrResult<SeqNode> completeSeq(
      std::shared_ptr<SeqNode> node,
      std::vector<PtrResult<StmtNode>> stmt_results) {
    return complete(node, stmt_results);
  }

  PtrResult<LetNode> completeLet(std::shared_ptr<LetNode> node,
                                 Result<VarDeclNode> var_decl_result,
                                 PtrResult<ExprNode> expr_result,
                                 PtrResult<StmtNode> scope_result) {
    return complete(node, var_decl_result, expr_result, scope_result);
  }

  PtrResult<AsgnNode> completeAsgn(std::shared_ptr<AsgnNode> node,
                                   Result<VarLocNode> var_loc_result,
                                   PtrResult<ExprNode> expr_result) {
    return complete(node, var_loc_result, expr_result);
  }

  PtrResult<LoopNode> completeLoop(
      std::shared_ptr<LoopNode> node,
      UniquePtrResult<InductionVarNode>&& induction_var_result,
      PtrResult<StmtNode> body_result) {
    RETURN_ORIGINAL_IF_UNCHANGED(PtrResult<LoopNode>, node,
                                 induction_var_result.changed, body_result);
    return PtrResult<LoopNode>(
        true, LoopNode::create(*(induction_var_result.node),
                               node->lower_bound(), node->upper_bound(),
                               node->stride(), extract(body_result)));
  }

  PtrResult<LoopNode> completeParLoop(
      std::shared_ptr<ParLoopNode> node,
      UniquePtrResult<InductionVarNode>&& induction_var_result,
      PtrResult<StmtNode> body_result) {
    RETURN_ORIGINAL_IF_UNCHANGED(PtrResult<ParLoopNode>, node,
                                 induction_var_result.changed, body_result);
    return PtrResult<LoopNode>(
        true, ParLoopNode::create(*(induction_var_result.node),
                                  node->lower_bound(), node->upper_bound(),
                                  node->stride(), extract(body_result)));
  }

  PtrResult<CriticalSectionNode> completeCriticalSection(
      std::shared_ptr<CriticalSectionNode> node,
      std::vector<PtrResult<AsgnNode>> critical_write_results) {
    return complete(node, critical_write_results);
  }

  Result<DefineNode> completeDefine(const DefineNode& node) {
    return Result<DefineNode>(false, node);
  }

  UniquePtrResult<ScalarVarNode> completeScalarVar(const ScalarVarNode& node) {
    return UniquePtrResult<ScalarVarNode>(
        false, std::make_unique<ScalarVarNode>(node));
  }

  UniquePtrResult<InductionVarNode> completeInductionVar(
      const InductionVarNode& node) {
    return UniquePtrResult<InductionVarNode>(
        false, std::make_unique<InductionVarNode>(node));
  }

  UniquePtrResult<TensorVarNode> completeTensorVar(const TensorVarNode& node) {
    return UniquePtrResult<TensorVarNode>(
        false, std::make_unique<TensorVarNode>(node));
  }

  Result<VarDeclNode> completeVarDecl(const VarDeclNode& node,
                                      UniquePtrResult<VarNode>&& var_result) {
    if (!var_result.changed) {
      return Result<VarDeclNode>(false, node);
    }

    return Result<VarDeclNode>(true, VarDeclNode(*(var_result.node)));
  }

  Result<VarLocNode> completeVarLoc(
      const VarLocNode& node, UniquePtrResult<VarRefNode>&& var_ref_result) {
    if (!var_ref_result.changed) {
      return Result<VarLocNode>(false, node);
    }

    return Result<VarLocNode>(true, *(var_ref_result.node));
  }

  UniquePtrResult<DefaultVarRefNode> completeDefaultVarRef(
      const DefaultVarRefNode& node, UniquePtrResult<VarNode>&& var_result) {
    // we are forced to make a copy here.
    return UniquePtrResult<DefaultVarRefNode>(
        var_result.changed,
        std::make_unique<DefaultVarRefNode>(*var_result.node));
  }

  UniquePtrResult<TensorVarRefNode> completeTensorVarRef(
      const TensorVarRefNode& node, UniquePtrResult<TensorVarNode>&& var_result,
      Result<IndexExpressionNode> index_expr_result) {
    // forced to copy.
    return UniquePtrResult<TensorVarRefNode>(
        var_result.changed, std::make_unique<TensorVarRefNode>(
                                *var_result.node, index_expr_result.node));
  }

  Result<IndexExpressionNode> completeIndexExpression(
      const IndexExpressionNode& node) {
    return Result<IndexExpressionNode>(false, node);
  }

 protected:
  // Helpers to determine whether any result has changed within our underlying
  // IR.
  static inline bool changed(bool result) { return result; }

  template <typename NodeT>
  static inline bool changed(const Result<NodeT>& result) {
    return result.changed;
  }

  template <typename NodeT>
  static inline bool changed(const UniquePtrResult<NodeT>& result) {
    return result.changed;
  }

  template <typename NodeT>
  static inline bool changed(const PtrResult<NodeT>& result) {
    return result.changed;
  }

  template <typename NodeT>
  static inline bool changed(const std::vector<Result<NodeT>>& results) {
    return std::accumulate(
        results.begin(), results.end(), false,
        [](bool b, const Result<NodeT>& y) { return b || y.changed; });
  }

  template <typename NodeT>
  static inline bool changed(const std::vector<PtrResult<NodeT>>& results) {
    return std::accumulate(
        results.begin(), results.end(), false,
        [](bool b, const PtrResult<NodeT>& y) { return b || y.changed; });
  }

  template <typename... ResultTs>
  static inline bool changed(ResultTs... results) {
    return (... || changed(results));
  }

  // Helpers to extract the underlying node from a `Result`.
  template <
      typename NodeT,
      std::enable_if_t<std::is_convertible_v<std::decay_t<NodeT>*, IrNode*>,
                       bool> = true>
  static std::vector<std::shared_ptr<NodeT>> extract(
      const std::vector<PtrResult<NodeT>>& results) {
    std::vector<std::shared_ptr<NodeT>> nodes;
    std::transform(results.begin(), results.end(), std::back_inserter(nodes),
                   [](const PtrResult<NodeT> result) { return result.node; });
    return nodes;
  }

  template <
      typename NodeT,
      std::enable_if_t<std::is_convertible_v<std::decay_t<NodeT>*, IrNode*>,
                       bool> = true>
  static std::vector<NodeT> extract(const std::vector<Result<NodeT>>& results) {
    std::vector<NodeT> nodes;
    std::transform(results.begin(), results.end(), std::back_inserter(nodes),
                   [](const Result<NodeT> result) { return result.node; });
    return nodes;
  }

  template <
      typename NodeT,
      std::enable_if_t<std::is_convertible_v<std::decay_t<NodeT>*, IrNode*>,
                       bool> = true>
  static std::shared_ptr<NodeT> extract(PtrResult<NodeT> result) {
    return result.node;
  }

  template <
      typename NodeT,
      std::enable_if_t<std::is_convertible_v<std::decay_t<NodeT>*, IrNode*>,
                       bool> = true>
  static NodeT extract(Result<NodeT> result) {
    return result.node;
  }

  // Generic templates to return a new node on change, or the original node if
  // unchanged.
  template <typename NodeT, typename... ResultTs>
  PtrResult<NodeT> complete(std::shared_ptr<NodeT> node, ResultTs... results) {
    bool did_change = (... || changed(results));
    if (!did_change) {
      return PtrResult<NodeT>(false, node);
    }

    return PtrResult<NodeT>(true, std::make_shared<NodeT>(extract(results)...));
  }

  template <typename NodeT, typename... ResultTs>
  Result<NodeT> complete(const NodeT& node, ResultTs... results) {
    bool did_change = (... || changed(results));
    if (!did_change) {
      return Result<NodeT>(false, node);
    }

    return Result<NodeT>(true, NodeT(extract(results)...));
  }

  inline VisitorT* visitor() { return static_cast<VisitorT*>(this); }
};

/*
 * Invokes vanilla functionality, which will return the original IR unchanged.
 */
class TrivialPathCopyVisitor : public PathCopyVisitor<TrivialPathCopyVisitor> {
 public:
  static FunctionNodePtr copy(FunctionNodePtr node) {
    TrivialPathCopyVisitor visitor;
    return extract(visitor.visit(node));
  }
};

}  // namespace peachyir

#undef DELEGATE
#undef RETURN_ORIGINAL_IF_UNCHANGED
