#include "peachy_ir_agent/ir.h"

#include <iostream>

namespace peachyir {

std::ostream& operator<<(std::ostream& os, IrNode::Kind kind) {
  switch (kind) {
    case IrNode::Kind::kUnknown:
      os << "Unknown";
      return os;
    case IrNode::Kind::kFunction:
      os << "Function";
      return os;
    case IrNode::Kind::kDefine:
      os << "Define";
      return os;
    case IrNode::Kind::kVar:
      os << "Var";
      return os;
    case IrNode::Kind::kScalarVar:
      os << "ScalarVar";
      return os;
    case IrNode::Kind::kInductionVar:
      os << "InductionVar";
      return os;
    case IrNode::Kind::kTensorVar:
      os << "TensorVar";
      return os;
    case IrNode::Kind::kVarDecl:
      os << "VarDecl";
      return os;
    case IrNode::Kind::kVarLoc:
      os << "VarLoc";
      return os;
    case IrNode::Kind::kVarRef:
      os << "VarRef";
      return os;
    case IrNode::Kind::kDefaultVarRef:
      os << "DefaultVarRef";
      return os;
    case IrNode::Kind::kTensorVarRef:
      os << "TensorVarRef";
      return os;
    case IrNode::Kind::kIndexExpression:
      os << "IndexExpression";
      return os;
    case IrNode::Kind::kExpr:
      os << "Expr";
      return os;
    case IrNode::Kind::kBinop:
      os << "Binop";
      return os;
    case IrNode::Kind::kUnop:
      os << "Unop";
      return os;
    case IrNode::Kind::kVarExpr:
      os << "VarExpr";
      return os;
    case IrNode::Kind::kStmt:
      os << "Stmt";
      return os;
    case IrNode::Kind::kSeq:
      os << "Seq";
      return os;
    case IrNode::Kind::kNop:
      os << "Nop";
      return os;
    case IrNode::Kind::kLet:
      os << "Let";
      return os;
    case IrNode::Kind::kAsgn:
      os << "Asgn";
      return os;
    case IrNode::Kind::kLoop:
      os << "Loop";
      return os;
    case IrNode::Kind::kParLoop:
      os << "ParLoop";
      return os;
    case IrNode::Kind::kCriticalSection:
      os << "CriticalSection";
      return os;
  }
}

std::ostream& operator<<(std::ostream& os, Type ty) {
  switch (ty) {
    case Type::kNat:
      os << "nat";
      return os;
    case Type::kFloat:
      os << "float";
      return os;
  }
}

std::ostream& operator<<(std::ostream& os, BinopNode::OpCode op) {
  switch (op) {
    case BinopNode::OpCode::kAdd:
      os << "+";
      return os;
    case BinopNode::OpCode::kSub:
      os << "-";
      return os;
    case BinopNode::OpCode::kMul:
      os << "*";
      return os;
    case BinopNode::OpCode::kDiv:
      os << "/";
      return os;
  }
}

std::ostream& operator<<(std::ostream& os, UnopNode::OpCode op) {
  switch (op) {
    case UnopNode::OpCode::kNeg:
      os << "-";
      return os;
  }
}

std::ostream& operator<<(std::ostream& os, LoopNode::Bound bound) {
  os << std::visit(
      [](auto&& b) -> std::string {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, VarExprNode>) {
          return b.var().name();
        } else {
          return std::to_string(b);
        }
      },
      bound);

  return os;
}

std::ostream& operator<<(std::ostream& os, LoopNode::Stride stride) {
  os << std::visit(
      [](auto&& s) -> std::string {
        using T = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<T, DefineNode>) {
          return s.name();
        } else {
          return std::to_string(s);
        }
      },
      stride);

  return os;
}

std::ostream& operator<<(std::ostream& os, const DefineNode& node) {
  os << "define " << node.name() << " = " << node.val();
  return os;
}

std::ostream& operator<<(std::ostream& os, const ScalarVarNode& node) {
  os << node.name() << ": " << node.type();
  return os;
}

std::ostream& operator<<(std::ostream& os, const InductionVarNode& node) {
  os << node.name() << ": axis(" << node.axis() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const TensorVarNode& node) {
  auto shape_str = [](const std::vector<size_t>& shape) {
    std::string s;

    s += "(";
    for (int i = 0; i < shape.size(); ++i) {
      s += std::to_string(shape[i]);
      if (i < shape.size() - 1) s += ", ";
    }
    s += ")";

    return s;
  };

  os << node.name() << ": " << shape_str(node.shape());
  return os;
}

std::ostream& operator<<(std::ostream& os, const IndexExpressionNode& node) {
  os << "[";
  for (int i = 0; i < node.axis_indices().size(); ++i) {
    const IndexExpressionNode::AxisIndex axis = node.axis_indices()[i];
    os << (axis.coeff == 1 ? "" : std::to_string(axis.coeff) + "*")
       << axis.var.name()
       << (axis.offset == 0 ? "" : " + " + std::to_string(axis.offset));
    if (i <= node.axis_indices().size() - 1) os << ", ";
  }
  os << "]";

  return os;
}

std::ostream& operator<<(std::ostream& os, const NopNode& node) { return os; }

}  // namespace peachyir
