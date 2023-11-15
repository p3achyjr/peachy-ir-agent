#pragma once

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#define DECLARE_FIELD(T, field) \
 private:                       \
  T field##_;                   \
                                \
 public:                        \
  const T& field() const { return field##_; }

#define DECLARE_PTR_FIELD(T, field)                     \
 private:                                               \
  std::shared_ptr<T> field##_;                          \
                                                        \
 public:                                                \
  std::shared_ptr<T> field() const { return field##_; } \
  T* raw_##field() const { return field##_.get(); }

// Use this to declare a potentially subtyped field that behaves with value
// semantics.
#define DECLARE_UNIQUE_PTR_FIELD(T, field) \
 private:                                  \
  std::unique_ptr<T> field##_;             \
                                           \
 public:                                   \
  const T& field() const { return *field##_; }

#define DECLARE_VEC_FIELD(T, field) \
 private:                           \
  std::vector<T> field##_;          \
                                    \
 public:                            \
  const std::vector<T>& field() const { return field##_; }

#define DECLARE_VEC_PTR_FIELD(T, field)     \
 private:                                   \
  std::vector<std::shared_ptr<T>> field##_; \
                                            \
 public:                                    \
  const std::vector<std::shared_ptr<T>>& field() const { return field##_; }

namespace peachyir {
class FunctionNode;
class DefineNode;
class VarNode;
class ScalarVarNode;
class InductionVarNode;
class TensorVarNode;
class VarRefNode;
class DefaultVarRef;
class TensorVarRefNode;
class StmtNode;
class VarDeclNode;
class ExprNode;
class BinopNode;
class UnopNode;
class VarExprNode;
class StmtNode;
class SeqNode;
class NopNode;
class LetNode;
class AsgnNode;
class LoopNode;
class ParLoopNode;
class CriticalSectionNode;

/*
 * (NOT CONCRETE) Base IrNode class.
 */
class IrNode {
 public:
  enum class Kind : uint8_t {
    //
    // Allowed in HLIR (IR fed to model).
    //
    kFunction = 0,
    kDefine,
    // Variables
    kScalarVar,
    kInductionVar,
    kTensorVar,
    kVarDecl,        // declaration of variable (with type, shape).
    kVarLoc,         // location of variable to assign to (i.e. variable write).
    kDefaultVarRef,  // Variable references, post-decl. Tensor variables must be
                     // indexed.
    kTensorVarRef,
    kIndexExpression,
    // Expressions.
    kBinop,
    kUnop,
    kVarExpr,  // Usage of variable in expression.
    // Statements.
    kSeq,
    kNop,
    kLet,
    kAsgn,
    kLoop,
    kParLoop,

    //
    // Allowed in LLIR (IR used for codegen).
    //
    kCriticalSection,
  };

  virtual ~IrNode() = default;
  inline Kind kind() const { return kind_; }

 protected:
  explicit IrNode(Kind kind) : kind_(kind) {}
  const Kind kind_;
};

enum class Type : uint8_t {
  kNat = 0,
  kFloat,
};

/*
 * Node representing a function. Contains `n` arguments, with types, `d`
 * defines, and a body `b`.
 */
using FunctionNodePtr = std::shared_ptr<FunctionNode>;
class FunctionNode : public IrNode {
 public:
  struct AxisInfo {
    size_t length;
    size_t tiles;
  };

  template <typename VarDeclT = VarDeclNode, typename DefineT = DefineNode,
            std::enable_if_t<
                std::is_convertible_v<std::decay_t<VarDeclT>*, VarDeclNode*> &&
                    std::is_convertible_v<std::decay_t<DefineT>*, DefineNode*>,
                bool> = true,
            typename VarDeclVec = std::vector<VarDeclT>,
            typename DefineVec = std::vector<DefineT>>
  FunctionNode(std::string name, VarDeclVec&& args, DefineVec&& defines,
               std::vector<AxisInfo> axes_info, std::shared_ptr<StmtNode> body)
      : IrNode(Kind::kFunction),
        name_(name),
        args_(args),
        defines_(defines),
        axes_info_(axes_info),
        body_(body) {}

  ~FunctionNode() override = default;

  DECLARE_FIELD(std::string, name);
  DECLARE_VEC_FIELD(VarDeclNode, args);
  DECLARE_VEC_FIELD(DefineNode, defines);
  DECLARE_VEC_FIELD(AxisInfo, axes_info);
  DECLARE_PTR_FIELD(StmtNode, body);

  inline size_t axis_len(size_t axis) const { return axes_info_[axis].length; }
  inline size_t axis_tiles(size_t axis) const { return axes_info_[axis].tiles; }

  template <typename VarDeclT = VarDeclNode, typename DefineT = DefineNode,
            std::enable_if_t<
                std::is_convertible_v<std::decay_t<VarDeclT>*, VarDeclNode*> &&
                    std::is_convertible_v<std::decay_t<DefineT>*, DefineNode*>,
                bool> = true,
            typename VarDeclVec = std::vector<VarDeclT>,
            typename DefineVec = std::vector<DefineT>>
  static FunctionNodePtr create(std::string name, VarDeclVec&& args,
                                DefineVec&& defines,
                                std::vector<AxisInfo> axes_info,
                                std::shared_ptr<StmtNode> body) {
    return std::make_shared<FunctionNode>(
        name, std::forward<std::vector<VarDeclT>>(args),
        std::forward<std::vector<DefineT>>(defines), axes_info, body);
  }
};

/*
 * Node representing a define. Defines a single constant.
 */
class DefineNode : public IrNode {
 public:
  DefineNode(std::string name, size_t val)
      : IrNode(Kind::kDefine), name_(name), val_(val) {}
  ~DefineNode() override = default;

  DECLARE_FIELD(std::string, name);
  DECLARE_FIELD(size_t, val);
};

/*
 * (NOT CONCRETE) Node representing a variable.
 */
class VarNode : public IrNode {
 public:
  ~VarNode() override = default;

  DECLARE_FIELD(std::string, name);

  virtual std::unique_ptr<VarNode> clone() const = 0;

 protected:
  VarNode(Kind kind, std::string name) : IrNode(kind), name_(name) {}
};

/*
 * Node representing a scalar variable.
 */
class ScalarVarNode : public VarNode {
 public:
  ScalarVarNode(std::string name, Type type)
      : VarNode(Kind::kScalarVar, name), type_(type) {}
  ~ScalarVarNode() override = default;

  DECLARE_FIELD(Type, type);

  std::unique_ptr<VarNode> clone() const override {
    return std::make_unique<ScalarVarNode>(*this);
  }
};

/*
 * Node representing a induction variable.
 */
class InductionVarNode : public VarNode {
 public:
  InductionVarNode(std::string name, size_t axis, size_t tile_level)
      : VarNode(Kind::kInductionVar, name),
        axis_(axis),
        tile_level_(tile_level) {}
  InductionVarNode(std::string name, size_t axis)
      : InductionVarNode(name, axis, -1) {}
  ~InductionVarNode() override = default;

  DECLARE_FIELD(size_t, axis);
  DECLARE_FIELD(size_t, tile_level);

  bool operator==(const InductionVarNode& ivar) const {
    return name() == ivar.name() && axis() == ivar.axis() &&
           tile_level() == ivar.tile_level();
  }

  bool operator!=(const InductionVarNode& ivar) const {
    return !(*this == ivar);
  }

  std::unique_ptr<VarNode> clone() const override {
    return std::make_unique<InductionVarNode>(*this);
  }
};

/*
 * Node representing a tensor variable.
 */
class TensorVarNode : public VarNode {
 public:
  TensorVarNode(std::string name, std::vector<size_t> shape)
      : VarNode(Kind::kTensorVar, name), shape_(shape) {}
  ~TensorVarNode() override = default;

  DECLARE_VEC_FIELD(size_t, shape);

  std::unique_ptr<VarNode> clone() const override {
    return std::make_unique<TensorVarNode>(*this);
  }
};

/*
 * (NOT CONCRETE) Node representing a variable reference.
 */
class VarRefNode : public IrNode {
 public:
  ~VarRefNode() override = default;

  DECLARE_FIELD(std::string, name);

  virtual std::unique_ptr<VarRefNode> clone() const = 0;

 protected:
  VarRefNode(Kind kind, std::string name) : IrNode(kind), name_(name) {}
};

class DefaultVarRefNode : public VarRefNode {
 public:
  template <
      typename VarT,
      std::enable_if_t<std::is_convertible_v<std::decay_t<VarT>*, VarNode*>,
                       bool> = true>
  DefaultVarRefNode(VarT&& var)
      : VarRefNode(Kind::kDefaultVarRef, var.name()), var_(var.clone()) {}
  DefaultVarRefNode(std::unique_ptr<VarNode>&& var)
      : VarRefNode(Kind::kDefaultVarRef, var->name()), var_(std::move(var)) {}
  ~DefaultVarRefNode() override = default;

  DECLARE_UNIQUE_PTR_FIELD(VarNode, var);

  std::unique_ptr<VarRefNode> clone() const override {
    return std::make_unique<DefaultVarRefNode>(var().clone());
  }
};

/*
 * Node representing an index into a tensor.
 */
class IndexExpressionNode : public IrNode {
 public:
  // Restrict index expressions to simple linear combinations of induction
  // variables.
  struct AxisIndex {
    int coeff;
    InductionVarNode var;
    int offset;

    AxisIndex(InductionVarNode var) : AxisIndex(1, var, 0) {}
    AxisIndex(int coeff, InductionVarNode var, int offset)
        : coeff(coeff), var(var), offset(offset) {}
  };

  IndexExpressionNode(std::vector<AxisIndex> axis_indices)
      : IrNode(Kind::kIndexExpression), axis_indices_(axis_indices) {}
  ~IndexExpressionNode() override = default;

  DECLARE_VEC_FIELD(AxisIndex, axis_indices);
};

/*
 * Node representing a tensor ref.
 */
class TensorVarRefNode : public VarRefNode {
 public:
  TensorVarRefNode(const TensorVarNode& var,
                   const IndexExpressionNode& index_expr)
      : VarRefNode(Kind::kTensorVarRef, var.name()),
        var_(var),
        index_expr_(index_expr) {}
  ~TensorVarRefNode() override = default;

  DECLARE_FIELD(TensorVarNode, var);
  DECLARE_FIELD(IndexExpressionNode, index_expr);

  std::unique_ptr<VarRefNode> clone() const override {
    return std::make_unique<TensorVarRefNode>(*this);
  }
};

/*
 * Node representing a variable we are declaring for the first time.
 */
class VarDeclNode : public IrNode {
 public:
  template <
      typename VarT,
      std::enable_if_t<std::is_convertible_v<std::decay_t<VarT>*, VarNode*>,
                       bool> = true>
  VarDeclNode(VarT&& var) : IrNode(Kind::kVarDecl), var_(var.clone()) {}
  ~VarDeclNode() override = default;

  VarDeclNode(const VarDeclNode& other)
      : IrNode(Kind::kVarDecl), var_(other.var().clone()) {}

  DECLARE_UNIQUE_PTR_FIELD(VarNode, var);
};

/*
 * Node representing a variable we are assigning to.
 */
class VarLocNode : public IrNode {
 public:
  template <typename VarRefT,
            std::enable_if_t<
                std::is_convertible_v<std::decay_t<VarRefT>*, VarRefNode*>,
                bool> = true>
  VarLocNode(VarRefT&& var_ref)
      : IrNode(Kind::kVarLoc), var_ref_(var_ref.clone()) {}
  ~VarLocNode() override = default;

  VarLocNode(const VarLocNode& other)
      : IrNode(Kind::kVarLoc), var_ref_(other.var_ref().clone()) {}

  DECLARE_UNIQUE_PTR_FIELD(VarRefNode, var_ref);
};

/*
 * (NOT CONCRETE) Node representing an expression.
 */
using ExprNodePtr = std::shared_ptr<ExprNode>;
class ExprNode : public IrNode {
 public:
  ~ExprNode() override = default;

 protected:
  ExprNode(Kind kind) : IrNode(kind) {}
};

/*
 * Node representing a binary expression.
 */
using BinopNodePtr = std::shared_ptr<BinopNode>;
class BinopNode : public ExprNode {
 public:
  enum class OpCode : uint8_t {
    kAdd = 0,
    kSub,
    kMul,
    kDiv,
  };

  BinopNode(std::shared_ptr<ExprNode> lhs, OpCode op,
            std::shared_ptr<ExprNode> rhs)
      : ExprNode(Kind::kBinop), lhs_(lhs), op_(op), rhs_(rhs) {}
  ~BinopNode() override = default;

  DECLARE_PTR_FIELD(ExprNode, lhs);
  DECLARE_FIELD(OpCode, op);
  DECLARE_PTR_FIELD(ExprNode, rhs);

  static BinopNodePtr create(std::shared_ptr<ExprNode> lhs, OpCode op,
                             std::shared_ptr<ExprNode> rhs) {
    return std::make_shared<BinopNode>(lhs, op, rhs);
  }
};

/*
 * Node representing a unary expression.
 */
using UnopNodePtr = std::shared_ptr<UnopNode>;
class UnopNode : public ExprNode {
 public:
  enum class OpCode : uint8_t {
    kNeg = 0,
  };

  UnopNode(OpCode op, std::shared_ptr<ExprNode> expr)
      : ExprNode(Kind::kUnop), op_(op), expr_(expr) {}
  ~UnopNode() override = default;

  DECLARE_FIELD(OpCode, op);
  DECLARE_PTR_FIELD(ExprNode, expr);

  static UnopNodePtr create(OpCode op, std::shared_ptr<ExprNode> expr) {
    return std::make_shared<UnopNode>(op, expr);
  }
};

/*
 * Node representing a reference to a variable (or a usage).
 */
using VarExprNodePtr = std::shared_ptr<VarExprNode>;
class VarExprNode : public ExprNode {
 public:
  template <typename VarRefT,
            std::enable_if_t<
                std::is_convertible_v<std::decay_t<VarRefT>*, VarRefNode*>,
                bool> = true>
  VarExprNode(VarRefT&& var_ref)
      : ExprNode(Kind::kVarExpr), var_ref_(var_ref.clone()) {}
  ~VarExprNode() override = default;

  VarExprNode(const VarExprNode& other)
      : ExprNode(Kind::kVarExpr), var_ref_(other.var_ref().clone()) {}

  DECLARE_UNIQUE_PTR_FIELD(VarRefNode, var_ref);

  template <typename VarRefT,
            std::enable_if_t<
                std::is_convertible_v<std::decay_t<VarRefT>*, VarRefNode*>,
                bool> = true>
  static VarExprNodePtr create(VarRefT&& var_ref) {
    return std::make_shared<VarExprNode>(var_ref);
  }
};

/*
 * (NOT CONCRETE) Node representing a statement.
 */
using StmtNodePtr = std::shared_ptr<StmtNode>;
class StmtNode : public IrNode {
 public:
  ~StmtNode() override = default;

 protected:
  StmtNode(Kind kind) : IrNode(kind) {}
};

/*
 * Node representing a sequence of statements.
 */
using SeqNodePtr = std::shared_ptr<SeqNode>;
class SeqNode : public StmtNode {
 public:
  SeqNode(const std::vector<std::shared_ptr<StmtNode>>& stmts)
      : StmtNode(Kind::kSeq), stmts_(stmts) {}
  SeqNode(std::vector<std::shared_ptr<StmtNode>>&& stmts)
      : StmtNode(Kind::kSeq), stmts_(stmts) {}
  ~SeqNode() override = default;

  DECLARE_VEC_PTR_FIELD(StmtNode, stmts);

  template <
      typename StmtT = StmtNode,
      std::enable_if_t<std::is_convertible_v<std::decay_t<StmtT>*, StmtNode*>,
                       bool> = true>
  static SeqNodePtr create(std::vector<std::shared_ptr<StmtT>>&& stmts) {
    return std::make_shared<StmtNode>(
        std::forward<std::vector<std::shared_ptr<StmtT>>>(stmts));
  }
};

/*
 * Node representing a no-op.
 */
using NopNodePtr = std::shared_ptr<NopNode>;
class NopNode : public StmtNode {
 public:
  NopNode() : StmtNode(Kind::kNop) {}
  ~NopNode() override = default;

  static NopNodePtr create() { return std::make_shared<NopNode>(); }
};

/*
 * Node representing a let-expression (as a statement).
 */
using LetNodePtr = std::shared_ptr<LetNode>;
class LetNode : public StmtNode {
 public:
  LetNode(const VarDeclNode& var_decl, std::shared_ptr<ExprNode> expr,
          std::shared_ptr<StmtNode> stmt)
      : StmtNode(Kind::kLet), var_decl_(var_decl), expr_(expr) {}
  LetNode(VarDeclNode&& var_decl, std::shared_ptr<ExprNode> expr,
          std::shared_ptr<StmtNode> stmt)
      : StmtNode(Kind::kLet), var_decl_(var_decl), expr_(expr) {}

  ~LetNode() override = default;

  DECLARE_FIELD(VarDeclNode, var_decl);
  DECLARE_PTR_FIELD(ExprNode, expr);
  DECLARE_PTR_FIELD(StmtNode, scope);

  static LetNodePtr create(const VarDeclNode& var_decl,
                           std::shared_ptr<ExprNode> expr,
                           std::shared_ptr<StmtNode> stmt) {
    return std::make_shared<LetNode>(var_decl, expr, stmt);
  }

  static LetNodePtr create(VarDeclNode&& var_decl,
                           std::shared_ptr<ExprNode> expr,
                           std::shared_ptr<StmtNode> stmt) {
    return std::make_shared<LetNode>(var_decl, expr, stmt);
  }
};

/*
 * Node representing assignment to a variable.
 */
using AsgnNodePtr = std::shared_ptr<AsgnNode>;
class AsgnNode : public StmtNode {
 public:
  AsgnNode(const VarLocNode& var_loc, std::shared_ptr<ExprNode> expr)
      : StmtNode(Kind::kAsgn), var_loc_(var_loc), expr_(expr) {}
  AsgnNode(VarLocNode&& var_loc, std::shared_ptr<ExprNode> expr)
      : StmtNode(Kind::kAsgn), var_loc_(var_loc), expr_(expr) {}
  ~AsgnNode() override = default;

  DECLARE_FIELD(VarLocNode, var_loc);
  DECLARE_PTR_FIELD(ExprNode, expr);

  static AsgnNodePtr create(const VarLocNode& var_loc,
                            std::shared_ptr<ExprNode> expr) {
    return std::make_shared<AsgnNode>(var_loc, expr);
  }

  static AsgnNodePtr create(VarLocNode&& var_loc,
                            std::shared_ptr<ExprNode> expr) {
    return std::make_shared<AsgnNode>(var_loc, expr);
  }
};

/*
 * Node representing a loop.
 */
using LoopNodePtr = std::shared_ptr<LoopNode>;
class LoopNode : public StmtNode {
 public:
  struct CompositeBound {
    using Coeff = std::variant<size_t, std::string>;
    Coeff coeff;
    InductionVarNode var;
    Coeff offset;

    CompositeBound(InductionVarNode var) : CompositeBound(1, var, 0) {}
    CompositeBound(Coeff coeff, InductionVarNode var, Coeff offset)
        : coeff(coeff), var(var), offset(offset) {}
  };

  using Bound = std::variant<size_t, CompositeBound>;
  using Stride =
      std::variant<size_t, std::string>;  // either inline stride, or
                                          // referencing a `DefineNode` key.

  LoopNode(const InductionVarNode& induction_var, Bound lower_bound,
           Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : LoopNode(Kind::kLoop, induction_var, lower_bound, upper_bound, stride,
                 body) {}
  LoopNode(InductionVarNode&& induction_var, Bound lower_bound,
           Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : LoopNode(Kind::kLoop, induction_var, lower_bound, upper_bound, stride,
                 body) {}

  ~LoopNode() override = default;

  DECLARE_FIELD(InductionVarNode, induction_var);
  DECLARE_FIELD(Bound, lower_bound);
  DECLARE_FIELD(Bound, upper_bound);
  DECLARE_FIELD(Stride, stride);
  DECLARE_PTR_FIELD(StmtNode, body);

  static LoopNodePtr create(const InductionVarNode& induction_var,
                            Bound lower_bound, Bound upper_bound, Stride stride,
                            std::shared_ptr<StmtNode> body) {
    return std::make_shared<LoopNode>(induction_var, lower_bound, upper_bound,
                                      stride, body);
  }

  static LoopNodePtr create(InductionVarNode&& induction_var, Bound lower_bound,
                            Bound upper_bound, Stride stride,
                            std::shared_ptr<StmtNode> body) {
    return std::make_shared<LoopNode>(induction_var, lower_bound, upper_bound,
                                      stride, body);
  }

 protected:
  template <typename InductionVarT = InductionVarNode,
            std::enable_if_t<std::is_convertible_v<std::decay_t<InductionVarT>*,
                                                   InductionVarNode*>,
                             bool> = true>
  LoopNode(Kind kind, InductionVarT&& induction_var, Bound lower_bound,
           Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : StmtNode(kind),
        induction_var_(std::forward<InductionVarT>(induction_var)),
        lower_bound_(lower_bound),
        upper_bound_(upper_bound),
        stride_(stride),
        body_(body) {}
};

/*
 * Node representing a parallel loop.
 */
using ParLoopNodePtr = std::shared_ptr<ParLoopNode>;
class ParLoopNode : public LoopNode {
 public:
  ParLoopNode(const InductionVarNode& induction_var, Bound lower_bound,
              Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : LoopNode(Kind::kParLoop, induction_var, lower_bound, upper_bound,
                 stride, body) {}
  ParLoopNode(InductionVarNode&& induction_var, Bound lower_bound,
              Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : LoopNode(Kind::kParLoop, induction_var, lower_bound, upper_bound,
                 stride, body) {}

  ~ParLoopNode() override = default;

  static ParLoopNodePtr create(const InductionVarNode& induction_var,
                               Bound lower_bound, Bound upper_bound,
                               Stride stride, std::shared_ptr<StmtNode> body) {
    return std::make_shared<ParLoopNode>(induction_var, lower_bound,
                                         upper_bound, stride, body);
  }

  static ParLoopNodePtr create(InductionVarNode&& induction_var,
                               Bound lower_bound, Bound upper_bound,
                               Stride stride, std::shared_ptr<StmtNode> body) {
    return std::make_shared<ParLoopNode>(induction_var, lower_bound,
                                         upper_bound, stride, body);
  }
};

/*
 * Node representing a critical section.
 */
using CriticalSectionNodePtr = std::shared_ptr<CriticalSectionNode>;
class CriticalSectionNode : public StmtNode {
 public:
  CriticalSectionNode(
      const std::vector<std::shared_ptr<AsgnNode>>& critical_writes)
      : StmtNode(Kind::kCriticalSection), critical_writes_(critical_writes) {}
  CriticalSectionNode(std::vector<std::shared_ptr<AsgnNode>>&& critical_writes)
      : StmtNode(Kind::kCriticalSection), critical_writes_(critical_writes) {}

  ~CriticalSectionNode() override = default;

  DECLARE_VEC_PTR_FIELD(AsgnNode, critical_writes);

  static CriticalSectionNodePtr create(
      const std::vector<std::shared_ptr<AsgnNode>>& critical_writes) {
    return std::make_shared<CriticalSectionNode>(critical_writes);
  }

  static CriticalSectionNodePtr create(
      std::vector<std::shared_ptr<AsgnNode>>&& critical_writes) {
    return std::make_shared<CriticalSectionNode>(critical_writes);
  }
};

// Stream operators for terminal node types.
std::ostream& operator<<(std::ostream& os, IrNode::Kind kind);
std::ostream& operator<<(std::ostream& os, Type ty);
std::ostream& operator<<(std::ostream& os, BinopNode::OpCode op);
std::ostream& operator<<(std::ostream& os, UnopNode::OpCode op);
std::ostream& operator<<(std::ostream& os, LoopNode::Bound bound);
std::ostream& operator<<(std::ostream& os, LoopNode::Stride stride);
std::ostream& operator<<(std::ostream& os, const DefineNode& node);
std::ostream& operator<<(std::ostream& os, const ScalarVarNode& node);
std::ostream& operator<<(std::ostream& os, const InductionVarNode& node);
std::ostream& operator<<(std::ostream& os, const TensorVarNode& node);
std::ostream& operator<<(std::ostream& os, const IndexExpressionNode& node);
std::ostream& operator<<(std::ostream& os, const NopNode& node);

// string cast for enums.
std::string str(IrNode::Kind kind);
std::string str(Type ty);
std::string str(BinopNode::OpCode op);
std::string str(UnopNode::OpCode op);
std::string str(LoopNode::Bound bound);
std::string str(LoopNode::Stride stride);

}  // namespace peachyir

#undef DECLARE_FIELD
#undef DECLARE_PTR_FIELD
#undef DECLARE_UNIQUE_PTR_FIELD
#undef DECLARE_VEC_FIELD
#undef DECLARE_VEC_PTR_FIELD
