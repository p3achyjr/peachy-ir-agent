#pragma once

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir.h"

namespace peachyir {

#define VISIT(METHOD, CLASS) \
  return visitor()->visit##METHOD(static_cast<const CLASS&>(node))

#define COMPLETE(METHOD, CLASS) \
  return visitor()->complete##METHOD(static_cast<const CLASS&>(node))

/*
 * Visitor Base Class.
 *
 * Based off https://llvm.org/doxygen/InstVisitor_8h_source.html
 *
 * We will define visit(NodeT& node) functions that contain preorder traversal
 * logic, and visit##Node(NodeT& node) functions that the user can register as
 * callbacks.
 *
 * The user can also override the visit##Node functions to contain traversal
 * logic, and dummy-out the visit$ functions.
 */
template <typename VisitorT>
class IrVisitor {
 public:
  // Generic visit method for vector fields.
  template <typename NodeT>
  void visit(const std::vector<NodeT>& v) {
    for (const NodeT& node : v) {
      visitor()->visit(node);
    }
  }

  // Delegation for pointer-based access.
  template <typename NodeT>
  void visit(std::shared_ptr<NodeT> node) {
    visitor()->visit(*node);
  }

  // Base visit methods that define traversal logic.
  void visit(const FunctionNode& node) {
    visitor()->visitFunction(node);
    visitor()->visit(node.args());
    visitor()->visit(node.defines());
    visitor()->visit(node.body());
    visitor()->completeFunction(node);
  }

  void visit(const DefineNode& node) {
    visitor()->visitDefine(node);
    visitor()->completeDefine(node);
  }

  void visit(const BinopNode& node) {
    visitor()->visitBinop(node);
    visitor()->visit(node.lhs());
    visitor()->visit(node.rhs());
    visitor()->completeBinop(node);
  }

  void visit(const UnopNode& node) {
    visitor()->visitUnop(node);
    visitor()->visit(node.expr());
    visitor()->completeUnop(node);
  }

  void visit(const VarExprNode& node) {
    visitor()->visitVarExpr(node);
    visitor()->visit(node.var_ref());
    visitor()->completeVarExpr(node);
  }

  void visit(const SeqNode& node) {
    visitor()->visitSeq(node);
    visit(node.stmts());
    visitor()->completeSeq(node);
  }

  void visit(const NopNode& node) {
    visitor()->visitNop(node);
    visitor()->completeNop(node);
  }

  void visit(const LetNode& node) {
    visitor()->visitLet(node);
    visitor()->visit(node.var_decl());
    visitor()->visit(node.expr());
    visitor()->visit(node.scope());
    visitor()->completeLet(node);
  }

  void visit(const AsgnNode& node) {
    visitor()->visitAsgn(node);
    visitor()->visit(node.var_loc());
    visitor()->visit(node.expr());
    visitor()->completeAsgn(node);
  }

  void visit(const LoopNode& node) {
    visitor()->visitLoop(node);
    visitor()->visit(node.induction_var());
    visitor()->visit(node.body());
    visitor()->completeLoop(node);
  }

  void visit(const ParLoopNode& node) {
    visitor()->visitParLoop(node);
    visitor()->visit(node.induction_var());
    visitor()->visit(node.body());
    visitor()->completeParLoop(node);
  }

  void visit(const CriticalSectionNode& node) {
    visitor()->visitCriticalSection(node);
    visitor()->visit(node.critical_writes());
    visitor()->completeCriticalSection(node);
  }

  void visit(const ScalarVarNode& node) {
    visitor()->visitScalarVar(node);
    visitor()->completeScalarVar(node);
  }

  void visit(const InductionVarNode& node) {
    visitor()->visitInductionVar(node);
    visitor()->completeInductionVar(node);
  }

  void visit(const TensorVarNode& node) {
    visitor()->visitTensorVar(node);
    visitor()->completeTensorVar(node);
  }

  void visit(const VarDeclNode& node) {
    visitor()->visitVarDecl(node);
    visitor()->visit(node.var());
    visitor()->completeVarDecl(node);
  }

  void visit(const VarLocNode& node) {
    visitor()->visitVarLoc(node);
    visitor()->visit(node.var_ref());
    visitor()->completeVarLoc(node);
  }

  void visit(const DefaultVarRefNode& node) {
    visitor()->visitDefaultVarRef(node);
    visitor()->visit(node.var());
    visitor()->completeDefaultVarRef(node);
  }

  void visit(const TensorVarRefNode& node) {
    visitor()->visitTensorVarRef(node);
    visitor()->visit(node.var());
    visitor()->visit(node.index_expr());
    visitor()->completeTensorVarRef(node);
  }

  void visit(const IndexExpressionNode& node) {
    visitor()->visitIndexExpression(node);
    visitor()->completeIndexExpression(node);
  }

  // Visit methods for subclassed types. Contains dispatch logic.
  void visit(const ExprNode& node) {
    switch (node.kind()) {
      case IrNode::Kind::kBinop:
        return visitor()->visit(static_cast<const BinopNode&>(node));
      case IrNode::Kind::kUnop:
        return visitor()->visit(static_cast<const UnopNode&>(node));
      case IrNode::Kind::kVarExpr:
        return visitor()->visit(static_cast<const VarExprNode&>(node));
      default:
        ABORT("Invalid ExprNode. OpCode: `%s`", str(node.kind()).c_str());
    }
  }

  void visit(const StmtNode& node) {
    switch (node.kind()) {
      case IrNode::Kind::kSeq:
        return visitor()->visit(static_cast<const SeqNode&>(node));
      case IrNode::Kind::kNop:
        return visitor()->visit(static_cast<const NopNode&>(node));
      case IrNode::Kind::kLet:
        return visitor()->visit(static_cast<const LetNode&>(node));
      case IrNode::Kind::kAsgn:
        return visitor()->visit(static_cast<const AsgnNode&>(node));
      case IrNode::Kind::kLoop:
        return visitor()->visit(static_cast<const LoopNode&>(node));
      case IrNode::Kind::kParLoop:
        return visitor()->visit(static_cast<const ParLoopNode&>(node));
      default:
        ABORT("Invalid StmtNode. OpCode: `%s`", str(node.kind()).c_str());
    }
  }

  void visit(const VarNode& node) {
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

  void visit(const VarRefNode& node) {
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
  void visitFunction(const FunctionNode& node) { VISIT(IrNode, IrNode); }
  void visitDefine(const DefineNode& node) { VISIT(IrNode, IrNode); }
  void visitVar(const VarNode& node) { VISIT(IrNode, IrNode); }
  void visitScalarVar(const ScalarVarNode& node) { VISIT(Var, VarNode); }
  void visitInductionVar(const InductionVarNode& node) { VISIT(Var, VarNode); }
  void visitTensorVar(const TensorVarNode& node) { VISIT(Var, VarNode); }
  void visitVarDecl(const VarDeclNode& node) { VISIT(IrNode, IrNode); }
  void visitVarLoc(const VarLocNode& node) { VISIT(IrNode, IrNode); }
  void visitVarRef(const VarRefNode& node) { VISIT(IrNode, IrNode); }
  void visitDefaultVarRef(const DefaultVarRefNode& node) {
    VISIT(VarRef, VarRefNode);
  }
  void visitTensorVarRef(const TensorVarRefNode& node) {
    VISIT(VarRef, VarRefNode);
  }
  void visitIndexExpression(const IndexExpressionNode& node) {
    VISIT(IrNode, IrNode);
  }
  void visitExpr(const ExprNode& node) { VISIT(IrNode, IrNode); }
  void visitBinop(const BinopNode& node) { VISIT(Expr, ExprNode); }
  void visitUnop(const UnopNode& node) { VISIT(Expr, ExprNode); }
  void visitVarExpr(const VarExprNode& node) { VISIT(Expr, ExprNode); }
  void visitStmt(const StmtNode& node) { VISIT(IrNode, IrNode); }
  void visitSeq(const SeqNode& node) { VISIT(Stmt, StmtNode); }
  void visitNop(const NopNode& node) { VISIT(Stmt, StmtNode); }
  void visitLet(const LetNode& node) { VISIT(Stmt, StmtNode); }
  void visitAsgn(const AsgnNode& node) { VISIT(Stmt, StmtNode); }
  void visitLoop(const LoopNode& node) { VISIT(Stmt, StmtNode); }
  void visitParLoop(const ParLoopNode& node) { VISIT(Stmt, StmtNode); }
  void visitCriticalSection(const CriticalSectionNode& node) {
    VISIT(Stmt, StmtNode);
  }

  // Default logic. Does nothing.
  void visitIrNode(const IrNode& node) {}

  // Post-order callbacks.
  void completeFunction(const FunctionNode& node) { COMPLETE(IrNode, IrNode); }
  void completeDefine(const DefineNode& node) { COMPLETE(IrNode, IrNode); }
  void completeVar(const VarNode& node) { COMPLETE(IrNode, IrNode); }
  void completeScalarVar(const ScalarVarNode& node) { COMPLETE(Var, VarNode); }
  void completeInductionVar(const InductionVarNode& node) {
    COMPLETE(Var, VarNode);
  }
  void completeTensorVar(const TensorVarNode& node) { COMPLETE(Var, VarNode); }
  void completeVarDecl(const VarDeclNode& node) { COMPLETE(IrNode, IrNode); }
  void completeVarLoc(const VarLocNode& node) { COMPLETE(IrNode, IrNode); }
  void completeVarRef(const VarRefNode& node) { COMPLETE(IrNode, IrNode); }
  void completeDefaultVarRef(const DefaultVarRefNode& node) {
    COMPLETE(VarRef, VarRefNode);
  }
  void completeTensorVarRef(const TensorVarRefNode& node) {
    COMPLETE(VarRef, VarRefNode);
  }
  void completeIndexExpression(const IndexExpressionNode& node) {
    COMPLETE(IrNode, IrNode);
  }
  void completeExpr(const ExprNode& node) { COMPLETE(IrNode, IrNode); }
  void completeBinop(const BinopNode& node) { COMPLETE(Expr, ExprNode); }
  void completeUnop(const UnopNode& node) { COMPLETE(Expr, ExprNode); }
  void completeVarExpr(const VarExprNode& node) { COMPLETE(Expr, ExprNode); }
  void completeStmt(const StmtNode& node) { COMPLETE(IrNode, IrNode); }
  void completeSeq(const SeqNode& node) { COMPLETE(Stmt, StmtNode); }
  void completeNop(const NopNode& node) { COMPLETE(Stmt, StmtNode); }
  void completeLet(const LetNode& node) { COMPLETE(Stmt, StmtNode); }
  void completeAsgn(const AsgnNode& node) { COMPLETE(Stmt, StmtNode); }
  void completeLoop(const LoopNode& node) { COMPLETE(Stmt, StmtNode); }
  void completeParLoop(const ParLoopNode& node) { COMPLETE(Stmt, StmtNode); }
  void completeCriticalSection(const CriticalSectionNode& node) {
    COMPLETE(Stmt, StmtNode);
  }

  // Default logic. Does nothing.
  void completeIrNode(const IrNode& node) {}

 protected:
  inline VisitorT* visitor() { return static_cast<VisitorT*>(this); }
};
}  // namespace peachyir

#undef VISIT
#undef COMPLETE
