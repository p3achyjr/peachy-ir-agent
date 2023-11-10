#pragma once

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir.h"

namespace peachyir {

#define DELEGATE(CLASS) \
  return visitor()->visit##CLASS(static_cast<const CLASS&>(node))

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
template <typename VisitorT, typename RetT = void>
class IrVisitor {
 public:
  // Generic visit method for vector fields.
  template <typename NodeT>
  void visit(const std::vector<NodeT>& v) {
    for (const NodeT& node : v) {
      visitor()->visit(node);
    }
  }

  // Base visit methods that define traversal logic.
  void visit(const FunctionNode& fn) {
    visitor()->visitFunction(fn);
    visit(fn.args());
    visit(fn.defines());
    visitor()->visit(*fn.body());
  }

  void visit(const BinopNode& b) {
    visitor()->visitBinop(b);
    visitor()->visit(*b.lhs());
    visitor()->visit(*b.rhs());
  }

  void visit(const UnopNode& u) {
    visitor()->visitUnop(u);
    visitor()->visit(*u.expr());
  }

  void visit(const VarRefNode& var_ref) {
    visitor()->visitVarRef(var_ref);
    visitor()->visit(*var_ref.var());
  }

  void visit(const SeqNode& seq) {
    visitor()->visitSeq(seq);
    visit(seq.stmts());
  }

  void visit(const NopNode& nop) { visitor()->visitNop(nop); }

  void visit(const LetNode& let) {
    visitor()->visitLet(let);
    visitor()->visit(*let.var_decl());
    visitor()->visit(*let.expr());
    visitor()->visit(*let.scope());
  }

  void visit(const AsgnNode& asgn) {
    visitor()->visitAsgn(asgn);
    visitor()->visit(*asgn.var_loc());
    visitor()->visit(*asgn.expr());
  }

  void visit(const LoopNode& loop) {
    visitor()->visitLoop(loop);
    visitor()->visit(*loop.induction_var());
    visitor()->visit(loop.body());
  }

  void visit(const ParLoopNode& ploop) {
    visitor()->visitParLoop(ploop);
    visitor()->visit(*ploop.induction_var());
    visitor()->visit(ploop.body());
  }

  void visit(const VarLocNode& var_loc) {
    visitor()->visitVarLoc(var_loc);
    visitor()->visit(*var_loc.var());
  }

  void visit(const VarDeclNode& var_decl) {
    visitor()->visitVarLoc(var_decl);
    visitor()->visit(*var_decl.var());
  }

  void visit(const CriticalSectionNode& cs) {
    visitor()->visitCriticalSection(cs);
    visit(cs.critical_writes());
  }

  RetT visit(const IrNode& node) {
    switch (node.kind()) {
      case IrNode::Kind::kFunction:
        return visitor()->visit(node);
      case IrNode::Kind::kDefine:
        return visitor()->visit(node);
      case IrNode::Kind::kExpr:
        return visitor()->visitExpr(node);
      case IrNode::Kind::kBinop:
        return visitor()->visit(node);
      case IrNode::Kind::kUnop:
        return visitor()->visit(node);
      case IrNode::Kind::kVarRef:
        return visitor()->visit(node);
      case IrNode::Kind::kStmt:
        return visitor()->visitStmt(node);
      case IrNode::Kind::kSeq:
        return visitor()->visit(node);
      case IrNode::Kind::kNop:
        return visitor()->visit(node);
      case IrNode::Kind::kLet:
        return visitor()->visit(node);
      case IrNode::Kind::kAsgn:
        return visitor()->visit(node);
      case IrNode::Kind::kLoop:
        return visitor()->visit(node);
      case IrNode::Kind::kParLoop:
        return visitor()->visit(node);
      case IrNode::Kind::kVarLoc:
        return visitor()->visit(node);
      case IrNode::Kind::kVarDecl:
        return visitor()->visit(node);
      case IrNode::Kind::kVar:
        return visitor()->visitVar(node);
      case IrNode::Kind::kScalarVar:
        return visitor()->visit(node);
      case IrNode::Kind::kInductionVar:
        return visitor()->visit(node);
      case IrNode::Kind::kTensorVar:
        return visitor()->visit(node);
      case IrNode::Kind::kIndexExpression:
        return visitor()->visit(node);
      case IrNode::Kind::kCriticalSection:
        return visitor()->visit(node);
      default:
        LOG(FATAL) << "Unknown Instruction (" << node.kind() << ").";
        std::exit(1);
    }
  }

  RetT visitFunction(const FunctionNode& node) { DELEGATE(IrNode); }
  RetT visitDefine(const DefineNode& node) { DELEGATE(IrNode); }
  RetT visitExpr(const ExprNode& node) { DELEGATE(IrNode); }
  RetT visitBinop(const BinopNode& node) { DELEGATE(ExprNode); }
  RetT visitUnop(const UnopNode& node) { DELEGATE(ExprNode); }
  RetT visitVarRef(const VarRefNode& node) { DELEGATE(ExprNode); }
  RetT visitStmt(const StmtNode& node) { DELEGATE(IrNode); }
  RetT visitSeq(const SeqNode& node) { DELEGATE(StmtNode); }
  RetT visitNop(const NopNode& node) { DELEGATE(StmtNode); }
  RetT visitLet(const LetNode& node) { DELEGATE(StmtNode); }
  RetT visitAsgn(const AsgnNode& node) { DELEGATE(StmtNode); }
  RetT visitLoop(const LoopNode& node) { DELEGATE(StmtNode); }
  RetT visitParLoop(const ParLoopNode& node) { DELEGATE(StmtNode); }
  RetT visitVarLoc(const VarLocNode& node) { DELEGATE(IrNode); }
  RetT visitVarDecl(const VarDeclNode& node) { DELEGATE(IrNode); }
  RetT visitVar(const VarNode& node) { DELEGATE(IrNode); }
  RetT visitScalarVar(const ScalarVarNode& node) { DELEGATE(VarNode); }
  RetT visitInductionVar(const InductionVarNode& node) { DELEGATE(VarNode); }
  RetT visitTensorVar(const TensorVarNode& node) { DELEGATE(VarNode); }
  RetT visitIndexExpression(const IndexExpressionNode& node) {
    DELEGATE(IrNode);
  }
  RetT visitCriticalSection(const CriticalSectionNode& node) {
    DELEGATE(StmtNode);
  }

  // Default logic. Does nothing.
  void visitIrNode(const IrNode& node) {}

 private:
  inline VisitorT* visitor() { return static_cast<VisitorT*>(this); }
};
}  // namespace peachyir
