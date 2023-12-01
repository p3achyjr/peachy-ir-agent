#include "peachy_ir_agent/codegen/cpp_gen.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir_visitor.h"

namespace peachyir {
namespace {

/*
 * Find parallel depth.
 */
class ParallelDepthFinder : public IrVisitor<ParallelDepthFinder> {
 public:
  using IrVisitor<ParallelDepthFinder>::visit;
  ParallelDepthFinder() : seen_parallel_(false), loop_depth_(0) {}
  ~ParallelDepthFinder() = default;

  void visitLoop(const LoopNode& node) {
    if (!seen_parallel_) {
      loop_depth_++;
    }

    seen_parallel_ = seen_parallel_ || node.is_parallel();
  }

  static int find(FunctionNodePtr ir) {
    ParallelDepthFinder visitor;
    visitor.visit(ir);
    return visitor.loop_depth_;
  }

 private:
  bool seen_parallel_;
  int loop_depth_;
};

/*
 * Generate C++ code from IR.
 */
class CppGenVisitor : public IrVisitor<CppGenVisitor> {
 public:
  using IrVisitor<CppGenVisitor>::visit;
  CppGenVisitor(int parallel_depth)
      : indent_depth_(0),
        loop_depth_(0),
        is_parallel_(false),
        parallel_depth_(parallel_depth) {}
  ~CppGenVisitor() = default;

  // Need to override traversal logic for these nodes.
  void visit(const FunctionNode& node) {
    is_parallel_ = node.is_parallel();
    ss_ << indents() << "void " << node.name() << " (";
    for (int i = 0; i < node.args().size(); ++i) {
      visit(node.args()[i]);
      if (i < node.args().size() - 1) {
        ss_ << ", ";
      }
    }
    ss_ << ") {\n";
    ++indent_depth_;

    for (const DefineNode& define : node.defines()) {
      ss_ << indents();
      visit(define);
      ss_ << ";\n";
    }

    visit(node.body());
    --indent_depth_;
    ss_ << "}\n";
  }

  void visit(const BinopNode& node) {
    visit(node.lhs());
    ss_ << " " << node.op() << " ";
    visit(node.rhs());
  }

  void visit(const LetNode& node) {
    ss_ << indents();
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

  void visit(const LoopNode& node) {
    if (loop_depth_ == 0 && is_parallel_ && parallel_depth_ == 1) {
      ss_ << "#pragma omp parallel for schedule(static)\n";
    } else if (loop_depth_ == 0 && is_parallel_) {
      ss_ << "#pragma omp parallel for collapse(" << parallel_depth_
          << ") schedule(static)\n";
    }

    std::string ivar_str = cc_str(node.induction_var());
    ss_ << indents();
    ss_ << "for (";
    ss_ << "size_t " << ivar_str << " = " << cc_str(node.lower_bound()) << "; ";
    ss_ << ivar_str << " < " << cc_str(node.upper_bound()) << "; ";
    ss_ << ivar_str << " += " << cc_str(node.stride());
    ss_ << ") {\n";
    ++indent_depth_;
    ++loop_depth_;
    visit(node.body());
    --loop_depth_;
    --indent_depth_;
    ss_ << indents() << "}\n";
  }

  void visit(const CriticalSectionNode& node) {
    ss_ << "#pragma omp critical\n";
    ss_ << indents() << "{\n";
    ++indent_depth_;
    visit(node.critical_writes());
    --indent_depth_;
    ss_ << indents() << "}\n";
  }

  void visit(const TensorVarRefNode& node) {
    visit(node.var());
    ss_ << cc_str(node.index_expr(), node.var().shape());
  }

  // Atomic visit functions.
  void visitDefine(const DefineNode& node) {
    ss_ << "static constexpr size_t " << node.name() << " = " << node.val();
  }

  void visitScalarVar(const ScalarVarNode& node) {
    if (current_scope_ == Scope::kDecl) {
      // we are declaring this variable for the first time, so print type.
      ss_ << cc_str(node.type()) << " " << node.name();
      return;
    }

    ss_ << node.name();
  }

  void visitInductionVar(const InductionVarNode& node) { ss_ << cc_str(node); }
  void visitTensorVar(const TensorVarNode& node) {
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

    if (current_scope_ == Scope::kDecl) {
      ss_ << "float* __restrict " << node.name() << " /* "
          << shape_str(node.shape()) << " */";
    } else {
      ss_ << node.name();
    }
  }

  void visitVarDecl(const VarDeclNode& node) { current_scope_ = Scope::kDecl; }
  void visitVarLoc(const VarLocNode& node) { current_scope_ = Scope::kLoc; }
  void visitVarExpr(const VarExprNode& node) { current_scope_ = Scope::kExpr; }
  void visitUnop(const UnopNode& node) { ss_ << node.op(); }
  void visitConst(const ConstNode& node) { ss_ << node; }

  static std::string gen(FunctionNodePtr ir, int parallel_depth) {
    CppGenVisitor visitor(parallel_depth);
    visitor.visit(ir);
    return visitor.ss_.str();
  }

 private:
  std::string cc_str(Type ty) {
    if (ty == Type::kFloat) {
      return "float";
    } else {
      return "size_t";
    }
  }

  std::string cc_str(const InductionVarNode& node) {
    std::stringstream ss;
    ss << node.name();
    if (node.tile_level() >= 0) {
      ss << "_T" << node.tile_level();
    }

    return ss.str();
  }

  std::string cc_str(const IndexExpressionNode& index_expr,
                     const std::vector<size_t>& shape) {
    // linear offset.
    size_t offset = 1;
    std::vector<size_t> offsets(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
      offsets[i] = offset;
      offset *= shape[i];
    }

    std::stringstream ss;
    ss << "[";
    for (int i = 0; i < index_expr.axis_indices().size(); ++i) {
      const IndexExpressionNode::AxisIndex& axis_index =
          index_expr.axis_indices()[i];
      if (offsets[i] != 1) ss << "(" << offsets[i] << " * ";
      if (axis_index.coeff == 1 && axis_index.offset == 0) {
        ss << cc_str(axis_index.var);
      } else {
        ss << "(";
        if (axis_index.coeff != 1) ss << axis_index.coeff << "*";
        ss << cc_str(axis_index.var);
        if (axis_index.offset != 0) ss << " + " << axis_index.offset;
        ss << ")";
      }
      if (offsets[i] != 1) ss << ")";

      if (i < index_expr.axis_indices().size() - 1) {
        ss << " + ";
      }
    }
    ss << "]";
    return ss.str();
  }

  std::string cc_str(std::variant<size_t, std::string> variant) {
    return std::visit(
        [](auto&& s) -> std::string {
          using T = std::decay_t<decltype(s)>;
          if constexpr (std::is_same_v<T, std::string>) {
            return s;
          } else {
            return std::to_string(s);
          }
        },
        variant);
  }

  std::string cc_str(const LoopNode::Bound& bound) {
    return std::visit(
        [this](auto&& b) -> std::string {
          using T = std::decay_t<decltype(b)>;
          if constexpr (std::is_same_v<T, LoopNode::CompositeBound>) {
            std::string coeff_str = cc_str(b.coeff);
            std::string var_str = cc_str(b.var);
            std::string offset_str = cc_str(b.offset);
            return (coeff_str == "1" ? "" : coeff_str + " * ") + var_str +
                   (offset_str == "0" ? "" : " + " + offset_str);
          } else {
            return std::to_string(b);
          }
        },
        bound);
  }

  std::string indents() { return std::string(kTabSize * indent_depth_, ' '); }

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
  int loop_depth_;
  bool is_parallel_;
  const int parallel_depth_;
};

}  // namespace

/* static */ std::string CppGen::apply(FunctionNodePtr ir) {
  int parallel_depth = ParallelDepthFinder::find(ir);
  return CppGenVisitor::gen(ir, parallel_depth);
}

/* static */ std::string CppAllocGen::apply(FunctionNodePtr ir) {
  std::stringstream ss;
  for (const VarDeclNode& var_decl : ir->args()) {
    if (var_decl.var().kind() != IrNode::Kind::kTensorVar) {
      continue;
    }

    const TensorVarNode& var =
        static_cast<const TensorVarNode&>(var_decl.var());
    const std::vector<size_t> shape = var.shape();
    const size_t buf_size = std::accumulate(shape.begin(), shape.end(), 1,
                                            std::multiplies<size_t>());

    ss << "float* " << var.name() << " = static_cast<float*>(aligned_alloc(32, "
       << buf_size * sizeof(float) << "));\n";
  }

  return ss.str();
}

/* static */ std::string CppInvokeGen::apply(FunctionNodePtr ir) {
  std::stringstream ss;
  ss << ir->name() << "(";
  for (int i = 0; i < ir->args().size(); ++i) {
    const VarDeclNode& var_decl = ir->args()[i];
    ss << var_decl.var().name();
    if (i < ir->args().size() - 1) ss << ", ";
  }
  ss << ");\n";

  return ss.str();
}

/* static */ std::string CppInitGen::apply(FunctionNodePtr ir) {
  std::stringstream ss;
  for (int i = 0; i < ir->args().size(); ++i) {
    const VarDeclNode& var_decl = ir->args()[i];
    if (var_decl.var().kind() != IrNode::Kind::kTensorVar) {
      continue;
    }

    const TensorVarNode& var =
        static_cast<const TensorVarNode&>(var_decl.var());
    const size_t buf_size = var.size();
    const std::string& name = var_decl.var().name();
    const std::string begin_it = name;
    const std::string end_it = name + " + " + std::to_string(buf_size);
    if (var_decl.is_dst()) {
      // is a write-to buffer.
      ss << "std::fill(" << begin_it << ", " << end_it << ", 0);\n";
    } else {
      // is a read-from buffer.
      ss << "std::transform(" << begin_it << ", " << end_it << ", " << begin_it
         << ", [](const float x) { return static_cast<float>(rand()) / "
            "RAND_MAX; });\n";
    }
  }

  return ss.str();
}

/* static */ std::string CppResetGen::apply(FunctionNodePtr ir) {
  std::stringstream ss;
  for (int i = 0; i < ir->args().size(); ++i) {
    const VarDeclNode& var_decl = ir->args()[i];
    if (var_decl.var().kind() != IrNode::Kind::kTensorVar) {
      continue;
    }

    const TensorVarNode& var =
        static_cast<const TensorVarNode&>(var_decl.var());
    const size_t buf_size = var.size();
    const std::string& name = var_decl.var().name();
    const std::string begin_it = name;
    const std::string end_it = name + " + " + std::to_string(buf_size);
    if (var_decl.is_dst()) {
      // is a write-to buffer.
      ss << "std::fill(" << begin_it << ", " << end_it << ", 0);\n";
    }

    // Do not reset read-from buffers.
  }

  return ss.str();
}
}  // namespace peachyir
