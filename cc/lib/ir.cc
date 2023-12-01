#include "peachy_ir_agent/ir.h"

#include <iostream>
#include <sstream>

namespace peachyir {

// Stream operators.
std::ostream& operator<<(std::ostream& os, IrNode::Kind kind) {
  switch (kind) {
    case IrNode::Kind::kFunction:
      os << "kFunction";
      break;
    case IrNode::Kind::kDefine:
      os << "kDefine";
      break;
    case IrNode::Kind::kScalarVar:
      os << "kScalarVar";
      break;
    case IrNode::Kind::kInductionVar:
      os << "kInductionVar";
      break;
    case IrNode::Kind::kTensorVar:
      os << "kTensorVar";
      break;
    case IrNode::Kind::kVarDecl:
      os << "kVarDecl";
      break;
    case IrNode::Kind::kVarLoc:
      os << "kVarLoc";
      break;
    case IrNode::Kind::kDefaultVarRef:
      os << "kDefaultVarRef";
      break;
    case IrNode::Kind::kTensorVarRef:
      os << "kTensorVarRef";
      break;
    case IrNode::Kind::kIndexExpression:
      os << "kIndexExpression";
      break;
    case IrNode::Kind::kBinop:
      os << "kBinop";
      break;
    case IrNode::Kind::kUnop:
      os << "kUnop";
      break;
    case IrNode::Kind::kVarExpr:
      os << "kVarExpr";
      break;
    case IrNode::Kind::kConst:
      os << "kConst";
      break;
    case IrNode::Kind::kSeq:
      os << "kSeq";
      break;
    case IrNode::Kind::kNop:
      os << "kNop";
      break;
    case IrNode::Kind::kLet:
      os << "kLet";
      break;
    case IrNode::Kind::kAsgn:
      os << "kAsgn";
      break;
    case IrNode::Kind::kLoop:
      os << "kLoop";
      break;
    case IrNode::Kind::kCriticalSection:
      os << "kCriticalSection";
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
        if constexpr (std::is_same_v<T, LoopNode::CompositeBound>) {
          std::string coeff_str = str(b.coeff);
          std::string var_str =
              b.var.name() + (b.var.tile_level() == -1
                                  ? ""
                                  : ".T" + std::to_string(b.var.tile_level()));
          std::string offset_str = str(b.offset);
          return (coeff_str == "1" ? "" : coeff_str + " * ") + var_str +
                 (offset_str == "0" ? "" : " + " + offset_str);
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
        if constexpr (std::is_same_v<T, std::string>) {
          return s;
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
  std::string name =
      node.tile_level() == -1
          ? node.name()
          : node.name() + ".T" + std::to_string(node.tile_level());
  os << name << ": axis(" << node.axis() << ")";
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

std::ostream& operator<<(std::ostream& os, const ConstNode& node) {
  if (node.type() == Type::kFloat) {
    os << node.val();
  } else {
    os << static_cast<size_t>(node.val());
  }

  return os;
}

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
