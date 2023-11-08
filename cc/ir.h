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

#define DECLARE_PTR_FIELD(T, field) \
 private:                           \
  std::unique_ptr<T> field##_;      \
                                    \
 public:                            \
  T* field() const { return field##_.get(); }

namespace ir {
class FunctionNode;
class DefineNode;
class ExprNode;
class BinopNode;
class UnopNode;
class VarRefNode;
class StmtNode;
class SeqNode;
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
  virtual Kind kind() = 0;
};

enum class Type : uint8_t {
  kInt = 0,
  kFloat,
};

/*
 * Node representing a function. Contains `n` arguments, with types, `d`
 * defines, and a body `b`.
 */
class FunctionNode : public IrNode {
 public:
  ~FunctionNode() override = default;
  inline Kind kind() override { return Kind::kFunction; }

  DECLARE_VEC_FIELD(VarDeclNode, args);
  DECLARE_VEC_FIELD(DefineNode, defines);
  DECLARE_PTR_FIELD(StmtNode, body);
};

/*
 * Node representing a define. Defines a single constant.
 */
class DefineNode : public IrNode {
 public:
  ~DefineNode() override = default;
  inline Kind kind() override { return Kind::kDefine; }

  DECLARE_FIELD(std::string, ident);
  DECLARE_FIELD(size_t, val);
};

/*
 * Node representing an expression.
 */
class ExprNode : public IrNode {
 public:
  ~ExprNode() override = default;
  inline Kind kind() override { return Kind::kExpr; }
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

  ~BinopNode() override = default;
  inline Kind kind() override { return Kind::kBinop; }

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

  ~UnopNode() override = default;
  inline Kind kind() override { return Kind::kUnop; }

  DECLARE_FIELD(OpCode, op);
  DECLARE_PTR_FIELD(ExprNode, expr);
};

/*
 * Node representing a reference to a variable (or a usage).
 */
class VarRefNode : public ExprNode {
 public:
  ~VarRefNode() override = default;
  inline Kind kind() override { return Kind::kVarRef; }

  DECLARE_PTR_FIELD(VarNode, var);
};

/*
 * Node representing a statement.
 */
class StmtNode : public IrNode {
 public:
  ~StmtNode() override = default;
  inline Kind kind() override { return Kind::kStmt; }
};

/*
 * Node representing a sequence of statements.
 */
class SeqNode : public StmtNode {
 public:
  ~SeqNode() override = default;
  inline Kind kind() override { return Kind::kSeq; }

  DECLARE_VEC_FIELD(StmtNode, stmts);
};

/*
 * Node representing a let-expression (as a statement).
 */
class LetNode : public StmtNode {
 public:
  ~LetNode() override = default;
  inline Kind kind() override { return Kind::kLet; }

  DECLARE_PTR_FIELD(VarDeclNode, var_decl);
  DECLARE_PTR_FIELD(ExprNode, expr);
  DECLARE_PTR_FIELD(StmtNode, scope);
};

/*
 * Node representing assignment to a variable.
 */
class AsgnNode : public StmtNode {
 public:
  ~AsgnNode() override = default;
  inline Kind kind() override { return Kind::kAsgn; }

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

  ~LoopNode() override = default;
  inline Kind kind() override { return Kind::kLoop; }

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
  ~ParLoopNode() override = default;
  inline Kind kind() override { return Kind::kParLoop; }
};

/*
 * Node representing a variable we are assigning to.
 */
class VarLocNode : public IrNode {
 public:
  ~VarLocNode() override = default;
  inline Kind kind() override { return Kind::kVarLoc; }

  DECLARE_PTR_FIELD(VarNode, var);
};

/*
 * Node representing a variable we are declaring for the first time.
 */
class VarDeclNode : public IrNode {
 public:
  ~VarDeclNode() override = default;
  inline Kind kind() override { return Kind::kVarDecl; }

  DECLARE_PTR_FIELD(VarNode, var);
};

/*
 * Node representing a variable.
 */
class VarNode : public IrNode {
 public:
  ~VarNode() override = default;
  inline Kind kind() override { return Kind::kVar; }

  DECLARE_FIELD(std::string, name);
};

/*
 * Node representing a scalar variable.
 */
class ScalarVarNode : public VarNode {
 public:
  ~ScalarVarNode() override = default;
  inline Kind kind() override { return Kind::kScalarVar; }

  DECLARE_FIELD(Type, type);
};

/*
 * Node representing a induction variable.
 */
class InductionVarNode : public VarNode {
 public:
  ~InductionVarNode() override = default;
  inline Kind kind() override { return Kind::kInductionVar; }

  DECLARE_FIELD(size_t, axis);
};

/*
 * Node representing a tensor variable.
 */
class TensorVarNode : public VarNode {
 public:
  ~TensorVarNode() override = default;
  inline Kind kind() override { return Kind::kTensorVar; }

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

  ~IndexExpressionNode() override = default;
  inline Kind kind() override { return Kind::kIndexExpression; }

  DECLARE_VEC_FIELD(AxisIndex, axis_indices);
};

/*
 * Node representing a critical section.
 */
class CriticalSectionNode : public StmtNode {
 public:
  ~CriticalSectionNode() override = default;
  inline Kind kind() override { return Kind::kCriticalSection; }

  DECLARE_VEC_FIELD(AsgnNode, critical_writes);
};

}  // namespace ir
