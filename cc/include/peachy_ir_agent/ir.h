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
class DefineNode;
class StmtNode;
class VarDeclNode;

/*
 * (NOT CONCRETE) Base IrNode class.
 */
class IrNode {
 public:
  enum class Kind : uint8_t {
    kUnknown = 0,

    //
    // Allowed in HLIR (IR fed to model).
    //
    kFunction,
    kDefine,
    // Variables
    kVar,
    kScalarVar,
    kInductionVar,
    kTensorVar,
    kVarDecl,  // declaration of variable (with type, shape).
    kVarLoc,   // location of variable to assign to (i.e. variable write).
    kVarRef,   // Variable reference, post-decl. Tensor variables must be
               // indexed.
    kDefaultVarRef,
    kTensorVarRef,
    kIndexExpression,
    // Expressions.
    kExpr,
    kBinop,
    kUnop,
    kVarExpr,  // Usage of variable in expression.
    // Statements.
    kStmt,
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
  virtual Kind kind() const = 0;

 protected:
  IrNode() = default;
};

enum class Type : uint8_t {
  kNat = 0,
  kFloat,
};

/*
 * Node representing a function. Contains `n` arguments, with types, `d`
 * defines, and a body `b`.
 */
class FunctionNode : public IrNode {
 public:
  // Reference.
  template <typename VarDeclT = VarDeclNode, typename DefineT = DefineNode,
            typename StmtT = StmtNode,
            std::enable_if_t<std::is_convertible_v<VarDeclT, VarDeclNode> &&
                                 std::is_convertible_v<DefineT, DefineNode> &&
                                 std::is_convertible_v<StmtT, StmtNode>,
                             bool> = true>
  FunctionNode(std::string name, std::vector<VarDeclT>&& args,
               std::vector<DefineT> defines, StmtT&& body)
      : FunctionNode(name, args, defines, std::make_shared<StmtT>(body)) {}

  // Shared Ptr.
  template <typename VarDeclT = VarDeclNode, typename DefineT = DefineNode,
            std::enable_if_t<std::is_convertible_v<VarDeclT, VarDeclNode> &&
                                 std::is_convertible_v<DefineT, DefineNode>,
                             bool> = true>
  FunctionNode(std::string name, std::vector<VarDeclT>&& args,
               std::vector<DefineT> defines, std::shared_ptr<StmtNode> body)
      : name_(name), args_(args), defines_(defines), body_(body) {}

  ~FunctionNode() override = default;
  inline Kind kind() const override { return Kind::kFunction; }

  DECLARE_FIELD(std::string, name);
  DECLARE_VEC_FIELD(VarDeclNode, args);
  DECLARE_VEC_FIELD(DefineNode, defines);
  DECLARE_PTR_FIELD(StmtNode, body);
};

/*
 * Node representing a define. Defines a single constant.
 */
class DefineNode : public IrNode {
 public:
  DefineNode(std::string name, size_t val) : name_(name), val_(val) {}
  ~DefineNode() override = default;
  inline Kind kind() const override { return Kind::kDefine; }

  DECLARE_FIELD(std::string, name);
  DECLARE_FIELD(size_t, val);
};

/*
 * (NOT CONCRETE) Node representing a variable.
 */
class VarNode : public IrNode {
 public:
  ~VarNode() override = default;
  inline Kind kind() const override { return Kind::kVar; }

  DECLARE_FIELD(std::string, name);

 protected:
  VarNode(std::string name) : name_(name) {}
};

/*
 * Node representing a scalar variable.
 */
class ScalarVarNode : public VarNode {
 public:
  ScalarVarNode(std::string name, Type type) : VarNode(name), type_(type) {}
  ~ScalarVarNode() override = default;
  inline Kind kind() const override { return Kind::kScalarVar; }

  DECLARE_FIELD(Type, type);
};

/*
 * Node representing a induction variable.
 */
class InductionVarNode : public VarNode {
 public:
  InductionVarNode(std::string name, size_t axis)
      : VarNode(name), axis_(axis) {}
  ~InductionVarNode() override = default;
  inline Kind kind() const override { return Kind::kInductionVar; }

  DECLARE_FIELD(size_t, axis);
};

/*
 * Node representing a tensor variable.
 */
class TensorVarNode : public VarNode {
 public:
  TensorVarNode(std::string name, std::vector<size_t> shape)
      : VarNode(name), shape_(shape) {}
  ~TensorVarNode() override = default;
  inline Kind kind() const override { return Kind::kTensorVar; }

  DECLARE_VEC_FIELD(size_t, shape);
};

/*
 * (NOT CONCRETE) Node representing a variable reference.
 */
class VarRefNode : public IrNode {
 public:
  ~VarRefNode() override = default;
  inline Kind kind() const override { return Kind::kVarRef; }

 protected:
  VarRefNode() = default;
};

class DefaultVarRefNode : public VarRefNode {
 public:
  DefaultVarRefNode(const VarNode& var) : var_(var) {}
  DefaultVarRefNode(VarNode&& var) : var_(var) {}
  ~DefaultVarRefNode() override = default;
  inline Kind kind() const override { return Kind::kDefaultVarRef; }

  DECLARE_FIELD(VarNode, var);
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
      : axis_indices_(axis_indices) {}
  ~IndexExpressionNode() override = default;
  inline Kind kind() const override { return Kind::kIndexExpression; }

  DECLARE_VEC_FIELD(AxisIndex, axis_indices);
};

/*
 * Node representing a tensor ref.
 */
class TensorVarRefNode : public VarRefNode {
 public:
  TensorVarRefNode(const TensorVarNode& var,
                   const IndexExpressionNode& index_expr)
      : var_(var), index_expr_(index_expr) {}
  ~TensorVarRefNode() override = default;
  inline Kind kind() const override { return Kind::kTensorVarRef; }

  DECLARE_FIELD(TensorVarNode, var);
  DECLARE_FIELD(IndexExpressionNode, index_expr);
};

/*
 * Node representing a variable we are declaring for the first time.
 */
class VarDeclNode : public IrNode {
 public:
  VarDeclNode(const VarNode& var) : var_(var) {}
  VarDeclNode(VarNode&& var) : var_(var) {}

  ~VarDeclNode() override = default;
  inline Kind kind() const override { return Kind::kVarDecl; }

  DECLARE_FIELD(VarNode, var);
};

/*
 * Node representing a variable we are assigning to.
 */
class VarLocNode : public IrNode {
 public:
  VarLocNode(const VarRefNode& var_ref) : var_ref_(var_ref) {}
  VarLocNode(VarRefNode&& var_ref) : var_ref_(var_ref) {}

  ~VarLocNode() override = default;
  inline Kind kind() const override { return Kind::kVarLoc; }

  DECLARE_FIELD(VarRefNode, var_ref);
};

/*
 * (NOT CONCRETE) Node representing an expression.
 */
class ExprNode : public IrNode {
 public:
  ~ExprNode() override = default;
  inline Kind kind() const override { return Kind::kExpr; }

 protected:
  ExprNode() = default;
};

/*
 * Node representing a binary expression.
 */
class BinopNode : public ExprNode {
 public:
  enum class OpCode : uint8_t {
    kAdd = 0,
    kSub,
    kMul,
    kDiv,
  };

  // Reference.
  template <typename LExprT = ExprNode, typename RExprT = ExprNode,
            std::enable_if_t<std::is_convertible_v<LExprT, ExprNode> &&
                                 std::is_convertible_v<RExprT, ExprNode>,
                             bool> = true>
  BinopNode(LExprT&& lhs, OpCode op, RExprT&& rhs)
      : lhs_(std::make_shared<LExprT>(std::forward<LExprT>(lhs))),
        op_(op),
        rhs_(std::make_shared<RExprT>(std::forward<RExprT>(rhs))) {}

  // Shared Ptr.
  BinopNode(std::shared_ptr<ExprNode> lhs, OpCode op,
            std::shared_ptr<ExprNode> rhs)
      : lhs_(lhs), op_(op), rhs_(rhs) {}
  ~BinopNode() override = default;
  inline Kind kind() const override { return Kind::kBinop; }

  DECLARE_PTR_FIELD(ExprNode, lhs);
  DECLARE_FIELD(OpCode, op);
  DECLARE_PTR_FIELD(ExprNode, rhs);
};

/*
 * Node representing a unary expression.
 */
class UnopNode : public ExprNode {
 public:
  enum class OpCode : uint8_t {
    kNeg = 0,
  };

  UnopNode(OpCode op, const ExprNode& expr)
      : op_(op), expr_(std::make_shared<ExprNode>(expr)) {}
  UnopNode(OpCode op, ExprNode&& expr)
      : op_(op), expr_(std::make_shared<ExprNode>(expr)) {}
  UnopNode(OpCode op, std::shared_ptr<ExprNode> expr) : op_(op), expr_(expr) {}
  ~UnopNode() override = default;
  inline Kind kind() const override { return Kind::kUnop; }

  DECLARE_FIELD(OpCode, op);
  DECLARE_PTR_FIELD(ExprNode, expr);
};

/*
 * Node representing a reference to a variable (or a usage).
 */
class VarExprNode : public ExprNode {
 public:
  VarExprNode(const VarRefNode& var_ref) : var_ref_(var_ref) {}
  VarExprNode(VarRefNode&& var_ref) : var_ref_(var_ref) {}

  ~VarExprNode() override = default;
  inline Kind kind() const override { return Kind::kVarExpr; }

  DECLARE_FIELD(VarRefNode, var_ref);
};

/*
 * (NOT CONCRETE) Node representing a statement.
 */
class StmtNode : public IrNode {
 public:
  ~StmtNode() override = default;
  inline Kind kind() const override { return Kind::kStmt; }

 protected:
  StmtNode() = default;
};

/*
 * Node representing a sequence of statements.
 */
class SeqNode : public StmtNode {
 public:
  SeqNode(const std::vector<std::shared_ptr<StmtNode>>& stmts)
      : stmts_(stmts) {}
  SeqNode(std::vector<std::shared_ptr<StmtNode>>&& stmts) : stmts_(stmts) {}
  ~SeqNode() override = default;
  inline Kind kind() const override { return Kind::kSeq; }

  DECLARE_VEC_PTR_FIELD(StmtNode, stmts);
};

/*
 * Node representing a no-op.
 */
class NopNode : public StmtNode {
 public:
  NopNode() = default;
  ~NopNode() override = default;
  inline Kind kind() const override { return Kind::kNop; }
};

/*
 * Node representing a let-expression (as a statement).
 */
class LetNode : public StmtNode {
 public:
  // Reference.
  template <typename VarDeclT = VarDeclNode, typename ExprT = ExprNode,
            typename StmtT = StmtNode,
            std::enable_if_t<std::is_convertible_v<VarDeclT, VarDeclNode> &&
                                 std::is_convertible_v<ExprT, ExprNode> &&
                                 std::is_convertible_v<StmtT, StmtNode>,
                             bool> = true>
  LetNode(VarDeclT&& var_decl, ExprT&& expr, StmtT&& stmt)
      : LetNode(std::forward<VarDeclT>(var_decl), std::make_shared<ExprT>(expr),
                std::make_shared<StmtT>(stmt)) {}

  // Shared Ptr.
  LetNode(const VarDeclNode& var_decl, std::shared_ptr<ExprNode> expr,
          std::shared_ptr<StmtNode> stmt)
      : var_decl_(var_decl), expr_(expr) {}
  LetNode(VarDeclNode&& var_decl, std::shared_ptr<ExprNode> expr,
          std::shared_ptr<StmtNode> stmt)
      : var_decl_(var_decl), expr_(expr) {}

  ~LetNode() override = default;
  inline Kind kind() const override { return Kind::kLet; }

  DECLARE_FIELD(VarDeclNode, var_decl);
  DECLARE_PTR_FIELD(ExprNode, expr);
  DECLARE_PTR_FIELD(StmtNode, scope);
};

/*
 * Node representing assignment to a variable.
 */
class AsgnNode : public StmtNode {
 public:
  // Reference.
  template <typename VarLocT = VarLocNode, typename ExprT = ExprNode,
            std::enable_if_t<std::is_convertible_v<VarLocT, VarLocNode> &&
                                 std::is_convertible_v<ExprT, ExprNode>,
                             bool> = true>
  AsgnNode(VarLocT&& var_loc, ExprT&& expr)
      : AsgnNode(std::forward<VarLocT>(var_loc),
                 std::make_shared<ExprT>(std::forward<ExprT>(expr))) {}

  // Shared Ptr.
  AsgnNode(const VarLocNode& var_loc, std::shared_ptr<ExprNode> expr)
      : var_loc_(var_loc), expr_(expr) {}
  AsgnNode(VarLocNode&& var_loc, std::shared_ptr<ExprNode> expr)
      : var_loc_(var_loc), expr_(expr) {}
  ~AsgnNode() override = default;
  inline Kind kind() const override { return Kind::kAsgn; }

  DECLARE_FIELD(VarLocNode, var_loc);
  DECLARE_PTR_FIELD(ExprNode, expr);
};

/*
 * Node representing a loop.
 */
class LoopNode : public StmtNode {
 public:
  using Bound = std::variant<size_t, VarExprNode>;
  using Stride = std::variant<size_t, DefineNode>;

  // Reference.
  template <
      typename InductionVarT = InductionVarNode, typename StmtT = StmtNode,
      std::enable_if_t<std::is_convertible_v<InductionVarT, InductionVarNode> &&
                           std::is_convertible_v<StmtT, StmtNode>,
                       bool> = true>
  LoopNode(InductionVarT&& induction_var, Bound lower_bound, Bound upper_bound,
           Stride stride, StmtT&& body)
      : LoopNode(std::forward<InductionVarT>(induction_var), lower_bound,
                 upper_bound, stride,
                 std::make_shared<StmtT>(std::forward<StmtT>(body))) {}

  // Shared Ptr.
  LoopNode(const InductionVarNode& induction_var, Bound lower_bound,
           Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : induction_var_(induction_var),
        lower_bound_(lower_bound),
        upper_bound_(upper_bound),
        body_(body) {}
  LoopNode(InductionVarNode&& induction_var, Bound lower_bound,
           Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : induction_var_(induction_var),
        lower_bound_(lower_bound),
        upper_bound_(upper_bound),
        body_(body) {}

  ~LoopNode() override = default;
  inline Kind kind() const override { return Kind::kLoop; }

  DECLARE_FIELD(InductionVarNode, induction_var);
  DECLARE_FIELD(Bound, lower_bound);
  DECLARE_FIELD(Bound, upper_bound);
  DECLARE_FIELD(Stride, stride);
  DECLARE_PTR_FIELD(StmtNode, body);
};

/*
 * Node representing a parallel loop.
 */
class ParLoopNode : public LoopNode {
 public:
  // Reference.
  template <
      typename InductionVarT = InductionVarNode, typename StmtT = StmtNode,
      std::enable_if_t<std::is_convertible_v<InductionVarT, InductionVarNode> &&
                           std::is_convertible_v<StmtT, StmtNode>,
                       bool> = true>
  ParLoopNode(InductionVarT&& induction_var, Bound lower_bound,
              Bound upper_bound, Stride stride, StmtT&& body)
      : LoopNode(std::forward<InductionVarT>(induction_var), lower_bound,
                 upper_bound, stride, std::forward<StmtT>(body)) {}

  // Shared Ptr.
  ParLoopNode(const InductionVarNode& induction_var, Bound lower_bound,
              Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : LoopNode(induction_var, lower_bound, upper_bound, stride, body) {}
  ParLoopNode(InductionVarNode&& induction_var, Bound lower_bound,
              Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : LoopNode(induction_var, lower_bound, upper_bound, stride, body) {}

  ~ParLoopNode() override = default;
  inline Kind kind() const override { return Kind::kParLoop; }
};

/*
 * Node representing a critical section.
 */
class CriticalSectionNode : public StmtNode {
 public:
  CriticalSectionNode(std::vector<std::shared_ptr<AsgnNode>> critical_writes)
      : critical_writes_(critical_writes) {}
  ~CriticalSectionNode() override = default;
  inline Kind kind() const override { return Kind::kCriticalSection; }

  DECLARE_VEC_PTR_FIELD(AsgnNode, critical_writes);
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

}  // namespace peachyir
