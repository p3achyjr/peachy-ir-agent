#include "peachy_ir_agent/codegen/critical_section.h"

#include <algorithm>
#include <optional>
#include <unordered_set>

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir_visitor.h"
#include "peachy_ir_agent/visitor/path_copy_visitor.h"

namespace peachyir {
namespace {

struct Vars {
  std::unordered_set<std::string> scalar_vars;
  std::unordered_set<std::string> tensor_vars;
};

class UnsynchronizedWriteFinder : public IrVisitor<UnsynchronizedWriteFinder> {
 public:
  UnsynchronizedWriteFinder()
      : parallel_induction_var_(std::nullopt), current_scope_(Scope::kNone) {}
  void visitLoop(const LoopNode& node) {
    if (node.is_parallel()) {
      parallel_induction_var_.emplace(node.induction_var());
    }
  }

  void completeLoop(const LoopNode& node) {
    if (node.is_parallel()) {
      parallel_induction_var_ = std::nullopt;
    }
  }

  void visitVarDecl(const VarDeclNode& node) { current_scope_ = Scope::kDecl; }
  void completeVarDecl(const VarDeclNode& node) {
    current_scope_ = Scope::kNone;
  }

  void visitVarLoc(const VarLocNode& node) { current_scope_ = Scope::kLoc; }
  void completeVarLoc(const VarLocNode& node) { current_scope_ = Scope::kNone; }

  void visitScalarVar(const ScalarVarNode& node) {
    if (current_scope_ == Scope::kNone) {
      return;
    }

    if (current_scope_ == Scope::kDecl && !parallel_induction_var_) {
      vars_outside_ploop_.scalar_vars.insert(node.name());
      return;
    }

    // We are in a write.
    if (vars_outside_ploop_.scalar_vars.find(node.name()) !=
        vars_outside_ploop_.scalar_vars.end()) {
      vars_require_sync_.scalar_vars.insert(node.name());
    }
  }

  void visitTensorVarRef(const TensorVarRefNode& node) {
    if (current_scope_ != Scope::kLoc) {
      return;
    }

    if (!parallel_induction_var_) {
      return;
    }

    // Check whether write depends on induction variable of parallel loop (i.e.
    // whether they are on the same axis).
    // If they are, then our writes move with the induction variable, so we do
    // not need guards.
    // If not, then different loop iterations may write to the same memory
    // region, and thus we need guards.
    size_t axis = parallel_induction_var_->axis();
    bool is_dependent = false;
    for (const auto& axis_index : node.index_expr().axis_indices()) {
      is_dependent = is_dependent || axis_index.var.axis() == axis;
    }

    if (!is_dependent) {
      // Write does move with respect with loop, so we can potentially share
      // writes across threads.
      vars_require_sync_.tensor_vars.insert(node.var().name());
    } else {
      // Writes move with respect to loop. Check if there are any conflicting
      // writes.
      for (const auto& index_expr : tensor_var_writes_[node.name()]) {
        if (index_expr != node.index_expr()) {
          vars_require_sync_.tensor_vars.insert(node.var().name());
          break;
        }
      }
    }

    // Add current write to set of writes.
    tensor_var_writes_[node.name()].emplace_back(node.index_expr());
  }

  static Vars findVarsNeedSync(FunctionNodePtr ir) {
    UnsynchronizedWriteFinder visitor;
    visitor.visit(ir);
    return visitor.vars_require_sync_;
  }

 private:
  enum class Scope : uint8_t {
    kNone = 0,
    kDecl = 1,
    kLoc = 2,
  };

  Vars vars_require_sync_;
  Vars vars_outside_ploop_;
  std::unordered_map<std::string, std::vector<IndexExpressionNode>>
      tensor_var_writes_;
  std::optional<InductionVarNode> parallel_induction_var_;
  Scope current_scope_;
};

class CriticalSectionRewriter
    : public PathCopyVisitor<CriticalSectionRewriter> {
 public:
  using PathCopyVisitor<CriticalSectionRewriter>::visit;
  CriticalSectionRewriter(const Vars& vars_need_sync)
      : vars_need_sync_(vars_need_sync) {}

  void visitDefaultVarRef(const DefaultVarRefNode& node) {
    if (vars_need_sync_.scalar_vars.find(node.name()) !=
        vars_need_sync_.scalar_vars.end()) {
      sync_next_asgn_ = true;
    }
  }

  void visitTensorVarRef(const TensorVarRefNode& node) {
    if (vars_need_sync_.tensor_vars.find(node.name()) !=
        vars_need_sync_.tensor_vars.end()) {
      sync_next_asgn_ = true;
    }
  }

  PtrResult<StmtNode> visit(std::shared_ptr<StmtNode> node) {
    if (node->kind() != IrNode::Kind::kAsgn) {
      return PathCopyVisitor<CriticalSectionRewriter>::visit(node);
    }

    // must be an asgn node.
    PtrResult<AsgnNode> asgn_result =
        PathCopyVisitor<CriticalSectionRewriter>::visit(
            std::static_pointer_cast<AsgnNode>(node));
    if (sync_next_asgn_) {
      PtrResult<CriticalSectionNode> result = PtrResult<CriticalSectionNode>(
          true, CriticalSectionNode::create({asgn_result.node}));
      sync_next_asgn_ = false;
      return result;
    }

    return asgn_result;
  }

  static FunctionNodePtr rewrite(FunctionNodePtr ir, Vars vars_need_sync) {
    CriticalSectionRewriter rewriter(vars_need_sync);
    PtrResult<FunctionNode> result = rewriter.visit(ir);
    return result.node;
  }

 private:
  const Vars vars_need_sync_;
  bool sync_next_asgn_;
};

}  // namespace

/* static */ FunctionNodePtr CriticalSectionTransform::apply(
    FunctionNodePtr ir) {
  if (!ir->is_parallel()) {
    return ir;
  }

  Vars unsynced_vars = UnsynchronizedWriteFinder::findVarsNeedSync(ir);
  if (unsynced_vars.scalar_vars.empty() && unsynced_vars.tensor_vars.empty()) {
    return ir;
  }

  return CriticalSectionRewriter::rewrite(ir, unsynced_vars);
}
}  // namespace peachyir
