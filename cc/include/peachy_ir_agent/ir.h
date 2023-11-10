#pragma once

#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#define DECLARE_VEC_FIELD(T, field) \
 private:                           \
  std::vector<T> field##_;          \
                                    \
 public:                            \
  const std::vector<T>& field() const { return field##_; }

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

namespace peachyir {
class FunctionNode;
class DefineNode;
class ExprNode;
class BinopNode;
class UnopNode;
class VarRefNode;
class StmtNode;
class SeqNode;
class NopNode;
class LetNode;
class AsgnNode;
class LoopNode;
class ParLoopNode;
class VarLocNode;
class VarDeclNode;
class VarNode;
class ScalarVarNode;
class InductionVarNode;
class TensorVarNode;
class IndexExpressionNode;
class CriticalSectionNode;

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
    // Expressions.
    kExpr,
    kBinop,
    kUnop,
    kVarRef,  // reference to variable (i.e. variable read).
    // Statements.
    kStmt,
    kSeq,
    kNop,
    kLet,
    kAsgn,
    kLoop,
    kParLoop,
    // Variables.
    kVarLoc,   // location of variable to assign to (i.e. variable write).
    kVarDecl,  // declaration of variable (with type, shape).
    kVar,
    kScalarVar,
    kInductionVar,
    kTensorVar,
    kIndexExpression,

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
  FunctionNode(std::vector<VarDeclT>&& args, std::vector<DefineT> defines,
               StmtT&& body)
      : FunctionNode(args, defines, std::make_shared<StmtT>(body)) {}

  // Shared Ptr.
  template <typename VarDeclT = VarDeclNode, typename DefineT = DefineNode,
            std::enable_if_t<std::is_convertible_v<VarDeclT, VarDeclNode> &&
                                 std::is_convertible_v<DefineT, DefineNode>,
                             bool> = true>
  FunctionNode(std::vector<VarDeclT>&& args, std::vector<DefineT> defines,
               std::shared_ptr<StmtNode> body)
      : args_(args), defines_(defines), body_(body) {}

  ~FunctionNode() override = default;
  inline Kind kind() const override { return Kind::kFunction; }

  DECLARE_VEC_FIELD(VarDeclNode, args);
  DECLARE_VEC_FIELD(DefineNode, defines);
  DECLARE_PTR_FIELD(StmtNode, body);
};

/*
 * Node representing a define. Defines a single constant.
 */
class DefineNode : public IrNode {
 public:
  DefineNode(std::string ident, size_t val) : ident_(ident), val_(val) {}
  ~DefineNode() override = default;
  inline Kind kind() const override { return Kind::kDefine; }

  DECLARE_FIELD(std::string, ident);
  DECLARE_FIELD(size_t, val);
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
  template <
      typename ExprT,
      std::enable_if_t<std::is_convertible_v<ExprT, ExprNode>, bool> = true>
  BinopNode(ExprT& lhs, OpCode op, ExprT&& rhs)
      : lhs_(std::make_shared<ExprT>(lhs)),
        op_(op),
        rhs_(std::make_shared<ExprT>(rhs)) {}

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
class VarRefNode : public ExprNode {
 public:
  VarRefNode(const VarNode& var) : var_(std::make_shared<VarNode>(var)) {}
  VarRefNode(VarNode&& var) : var_(std::make_shared<VarNode>(var)) {}
  VarRefNode(std::shared_ptr<VarNode> var) : var_(var) {}

  ~VarRefNode() override = default;
  inline Kind kind() const override { return Kind::kVarRef; }

  DECLARE_PTR_FIELD(VarNode, var);
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
  SeqNode(const std::vector<StmtNode>& stmts) : stmts_(stmts) {}
  SeqNode(std::vector<StmtNode>&& stmts) : stmts_(stmts) {}
  ~SeqNode() override = default;
  inline Kind kind() const override { return Kind::kSeq; }

  DECLARE_VEC_FIELD(StmtNode, stmts);
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
  template <typename VarDeclT, typename ExprT, typename StmtT,
            std::enable_if_t<std::is_convertible_v<VarDeclT, VarLocNode> &&
                                 std::is_convertible_v<ExprT, ExprNode> &&
                                 std::is_convertible_v<StmtT, StmtNode>,
                             bool> = true>
  LetNode(VarDeclT&& var_decl, ExprT&& expr, StmtT&& stmt)
      : LetNode(std::make_shared<VarDeclT>(var_decl),
                std::make_shared<ExprT>(expr), std::make_shared<StmtT>(stmt)) {}

  // Shared Ptr.
  LetNode(std::shared_ptr<VarDeclNode> var_decl, std::shared_ptr<ExprNode> expr,
          std::shared_ptr<StmtNode> stmt)
      : var_decl_(var_decl), expr_(expr) {}

  ~LetNode() override = default;
  inline Kind kind() const override { return Kind::kLet; }

  DECLARE_PTR_FIELD(VarDeclNode, var_decl);
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
  AsgnNode(VarLocNode&& var_loc, ExprNode&& expr)
      : AsgnNode(std::make_shared<VarLocT>(var_loc),
                 std::make_shared<ExprT>(expr)) {}

  // Shared Ptr.
  AsgnNode(std::shared_ptr<VarLocNode> var_loc, std::shared_ptr<ExprNode> expr)
      : var_loc_(var_loc), expr_(expr) {}
  ~AsgnNode() override = default;
  inline Kind kind() const override { return Kind::kAsgn; }

  DECLARE_PTR_FIELD(VarLocNode, var_loc);
  DECLARE_PTR_FIELD(ExprNode, expr);
};

/*
 * Node representing a loop.
 */
class LoopNode : public StmtNode {
 public:
  using Bound = std::variant<size_t, VarRefNode>;
  using Stride = std::variant<size_t, DefineNode>;

  // Reference.
  template <
      typename InductionVarT, typename StmtT,
      std::enable_if_t<std::is_convertible_v<InductionVarT, InductionVarNode> &&
                           std::is_convertible_v<StmtT, StmtNode>,
                       bool> = true>
  LoopNode(InductionVarT&& induction_var, Bound lower_bound, Bound upper_bound,
           Stride stride, StmtNode&& body)
      : LoopNode(std::make_shared<InductionVarNode>(induction_var), lower_bound,
                 upper_bound, stride, std::make_shared<StmtNode>(body)) {}

  // Shared Ptr.
  LoopNode(std::shared_ptr<InductionVarNode> induction_var, Bound lower_bound,
           Bound upper_bound, Stride stride, std::shared_ptr<StmtNode> body)
      : induction_var_(induction_var),
        lower_bound_(lower_bound),
        upper_bound_(upper_bound),
        body_(body) {}

  ~LoopNode() override = default;
  inline Kind kind() const override { return Kind::kLoop; }

  DECLARE_PTR_FIELD(InductionVarNode, induction_var);
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
      typename InductionVarT, typename StmtT,
      std::enable_if_t<std::is_convertible_v<InductionVarT, InductionVarNode> &&
                           std::is_convertible_v<StmtT, StmtNode>,
                       bool> = true>
  ParLoopNode(InductionVarT&& induction_var, Bound lower_bound,
              Bound upper_bound, Stride stride, StmtT&& body)
      : LoopNode(std::forward<InductionVarT>(induction_var), lower_bound,
                 upper_bound, stride, std::forward<StmtT>(body)) {}

  // Shared Ptr.
  ParLoopNode(std::shared_ptr<InductionVarNode> induction_var,
              Bound lower_bound, Bound upper_bound, Stride stride,
              std::shared_ptr<StmtNode> body)
      : LoopNode(induction_var, lower_bound, upper_bound, stride, body) {}

  ~ParLoopNode() override = default;
  inline Kind kind() const override { return Kind::kParLoop; }
};

/*
 * Node representing a variable we are assigning to.
 */
class VarLocNode : public IrNode {
 public:
  VarLocNode(const VarNode& var) : var_(std::make_shared<VarNode>(var)) {}
  VarLocNode(VarNode&& var) : var_(std::make_shared<VarNode>(var)) {}
  VarLocNode(std::shared_ptr<VarNode> var) : var_(var) {}

  ~VarLocNode() override = default;
  inline Kind kind() const override { return Kind::kVarLoc; }

  DECLARE_PTR_FIELD(VarNode, var);
};

/*
 * Node representing a variable we are declaring for the first time.
 */
class VarDeclNode : public IrNode {
 public:
  VarDeclNode(const VarNode& var) : var_(std::make_shared<VarNode>(var)) {}
  VarDeclNode(VarNode&& var) : var_(std::make_shared<VarNode>(var)) {}
  VarDeclNode(std::shared_ptr<VarNode> var) : var_(var) {}

  ~VarDeclNode() override = default;
  inline Kind kind() const override { return Kind::kVarDecl; }

  DECLARE_PTR_FIELD(VarNode, var);
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
  };

  IndexExpressionNode(std::vector<AxisIndex> axis_indices)
      : axis_indices_(axis_indices) {}
  ~IndexExpressionNode() override = default;
  inline Kind kind() const override { return Kind::kIndexExpression; }

  DECLARE_VEC_FIELD(AxisIndex, axis_indices);
};

/*
 * Node representing a critical section.
 */
class CriticalSectionNode : public StmtNode {
 public:
  CriticalSectionNode(std::vector<AsgnNode> critical_writes)
      : critical_writes_(critical_writes) {}
  ~CriticalSectionNode() override = default;
  inline Kind kind() const override { return Kind::kCriticalSection; }

  DECLARE_VEC_FIELD(AsgnNode, critical_writes);
};

}  // namespace peachyir
