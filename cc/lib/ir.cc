#include "peachy_ir_agent/ir.h"

#include <iostream>
#include <sstream>

namespace peachyir {

// Stream operators.
std::ostream& operator<<(std::ostream& os, IrNode::Kind kind) {
  switch (kind) {
    case IrNode::Kind::kFunction:
      os << "Function";
      break;
    case IrNode::Kind::kDefine:
      os << "Define";
      break;
    case IrNode::Kind::kScalarVar:
      os << "ScalarVar";
      break;
    case IrNode::Kind::kInductionVar:
      os << "InductionVar";
      break;
    case IrNode::Kind::kTensorVar:
      os << "TensorVar";
      break;
    case IrNode::Kind::kVarDecl:
      os << "VarDecl";
      break;
    case IrNode::Kind::kVarLoc:
      os << "VarLoc";
      break;
    case IrNode::Kind::kDefaultVarRef:
      os << "DefaultVarRef";
      break;
    case IrNode::Kind::kTensorVarRef:
      os << "TensorVarRef";
      break;
    case IrNode::Kind::kIndexExpression:
      os << "IndexExpression";
      break;
    case IrNode::Kind::kBinop:
      os << "Binop";
      break;
    case IrNode::Kind::kUnop:
      os << "Unop";
      break;
    case IrNode::Kind::kVarExpr:
      os << "VarExpr";
      break;
    case IrNode::Kind::kSeq:
      os << "Seq";
      break;
    case IrNode::Kind::kNop:
      os << "Nop";
      break;
    case IrNode::Kind::kLet:
      os << "Let";
      break;
    case IrNode::Kind::kAsgn:
      os << "Asgn";
      break;
    case IrNode::Kind::kLoop:
      os << "Loop";
      break;
    case IrNode::Kind::kParLoop:
      os << "ParLoop";
      break;
    case IrNode::Kind::kCriticalSection:
      os << "CriticalSection";
      break;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, Type ty) {
  switch (ty) {
    case Type::kNat:
      os << "nat";
      break;
    case Type::kFloat:
      os << "float";
      break;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, BinopNode::OpCode op) {
  switch (op) {
    case BinopNode::OpCode::kAdd:
      os << "+";
      break;
    case BinopNode::OpCode::kSub:
      os << "-";
      break;
    case BinopNode::OpCode::kMul:
      os << "*";
      break;
    case BinopNode::OpCode::kDiv:
      os << "/";
      break;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, UnopNode::OpCode op) {
  switch (op) {
    case UnopNode::OpCode::kNeg:
      os << "-";
      break;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, LoopNode::Bound bound) {
  os << std::visit(
      [](auto&& b) -> std::string {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, VarExprNode>) {
          return b.var_ref()->name();
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
    if (i < node.axis_indices().size() - 1) os << ", ";
  }
  os << "]";

  return os;
}

std::ostream& operator<<(std::ostream& os, const NopNode& node) { return os; }

// String functions.

namespace {
template <typename T>
std::string createString(T x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}
}  // namespace

std::string str(IrNode::Kind kind) { return createString(kind); }
std::string str(Type ty) { return createString(ty); }
std::string str(BinopNode::OpCode op) { return createString(op); }
std::string str(UnopNode::OpCode op) { return createString(op); }
std::string str(LoopNode::Bound bound) { return createString(bound); }
std::string str(LoopNode::Stride stride) { return createString(stride); }

}  // namespace peachyir
