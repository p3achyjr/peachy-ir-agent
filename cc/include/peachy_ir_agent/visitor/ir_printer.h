#pragma once

#include <numeric>

#include "peachy_ir_agent/ir_visitor.h"

#define PRINT_NAME_AND_RETURN_IF_IN_EXPR(var) \
  do {                                        \
    if (current_scope_ == Scope::kExpr) {     \
      ss_ << var.name();                      \
      return;                                 \
    }                                         \
  } while (0);

namespace peachyir {

class IrPrinter : public IrVisitor<IrPrinter> {
 public:
  using IrVisitor<IrPrinter>::visit;

  IrPrinter() : indent_depth_(0) {}
  ~IrPrinter() = default;

  template <typename NodeT>
  static std::string print(std::shared_ptr<NodeT> node) {
    IrPrinter printer;
    printer.visit(*node);
    return printer.str();
  }

  template <typename NodeT>
  static std::string print(const NodeT& node) {
    IrPrinter printer;
    printer.visit(node);
    return printer.str();
  }

  // Need to override traversal logic for these nodes.
  void visit(const FunctionNode& node) {
    ss_ << indents() << "def " << node.name() << " (";
    for (int i = 0; i < node.args().size(); ++i) {
      visit(node.args()[i]);
      if (i < node.args().size() - 1) {
        ss_ << ", ";
      }
    }
    ss_ << "):\n";
    ++indent_depth_;

    for (const DefineNode& define : node.defines()) {
      ss_ << indents();
      visit(define);
      ss_ << ";\n";
    }

    visit(node.body());
    --indent_depth_;
  }

  void visit(const BinopNode& node) {
    visit(node.lhs());
    ss_ << " " << node.op() << " ";
    visit(node.rhs());
  }

  void visit(const LetNode& node) {
    ss_ << indents() << "let ";
    visit(node.var_decl());
    ss_ << " = ";
    visit(node.expr());
    ss_ << ";\n";
    visit(node.scope());
  }

  void visit(const AsgnNode& node) {
    ss_ << indents();
    visit(node.var_loc());
    ss_ << " = ";
    visit(node.expr());
    ss_ << ";\n";
  }

  // Atomic visit functions.
  void visitDefine(const DefineNode& node) { ss_ << node; }

  void visitScalarVar(const ScalarVarNode& node) {
    PRINT_NAME_AND_RETURN_IF_IN_EXPR(node);
    ss_ << node;
  }

  void visitInductionVar(const InductionVarNode& node) {
    PRINT_NAME_AND_RETURN_IF_IN_EXPR(node);
    // We should never write to induction variables. Reaching here should be
    // an error, but we will just not print anything.
  }

  void visitTensorVar(const TensorVarNode& node) {
    if (current_scope_ == Scope::kDecl) {
      ss_ << node;
    } else {
      ss_ << node.name();
    }
  }

  void visitVarDecl(const VarDeclNode& node) { current_scope_ = Scope::kDecl; }
  void visitVarLoc(const VarLocNode& node) { current_scope_ = Scope::kLoc; }
  void visitIndexExpression(const IndexExpressionNode& node) { ss_ << node; }
  void visitUnop(const UnopNode& node) { ss_ << node.op(); }
  void visitVarExpr(const VarExprNode& node) { current_scope_ = Scope::kExpr; }
  void visitLoop(const LoopNode& node) {
    ss_ << indents() << "loop " << node.induction_var() << " in ("
        << node.lower_bound() << ", " << node.upper_bound() << ") "
        << node.stride() << ":\n";
    ++indent_depth_;
  }

  void visitParLoop(const ParLoopNode& node) {
    ss_ << indents() << "parallel " << node.induction_var() << " in ("
        << node.lower_bound() << ", " << node.upper_bound() << ") "
        << node.stride() << ":\n";
    ++indent_depth_;
  }

  void visitCriticalSection(const CriticalSectionNode& node) {
    ss_ << indents() << "critical:\n";
    ++indent_depth_;
  }

  void completeFunction(const FunctionNode& node) { --indent_depth_; }
  void completeLoop(const LoopNode& node) { --indent_depth_; }
  void completeParLoop(const ParLoopNode& node) { --indent_depth_; }
  void completeCriticalSection(const CriticalSectionNode& node) {
    --indent_depth_;
  }

 private:
  std::string indents() { return std::string(kTabSize * indent_depth_, ' '); }
  std::string str() { return ss_.str(); }

  enum class Scope : uint8_t {
    kNone = 0,
    kDecl = 1,
    kLoc = 2,
    kExpr = 3,
  };

  static constexpr int kTabSize = 2;
  std::stringstream ss_;
  int indent_depth_;
  Scope current_scope_;
};

}  // namespace peachyir
