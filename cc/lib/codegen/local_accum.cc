#include "peachy_ir_agent/codegen/local_accum.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir_visitor.h"
#include "peachy_ir_agent/visitor/ir_printer.h"
#include "peachy_ir_agent/visitor/path_copy_visitor.h"

namespace peachyir {
namespace {

/*
 * Finds all tensor writes that need local accumulators.
 *
 * A tensor write T requires a local accumulator if the immediately
 * surrounding loop's induction variable is not referenced in T's index
 * expression.
 *
 * Essentially what this means is that we are writing to the same
 * memory location in each iteration of the surrounding loop, so we can write to
 * a register instead.
 */
class AccumWriteFinder : public IrVisitor<AccumWriteFinder> {
 public:
  void visitLoop(const LoopNode& node) {
    ivar_stack_.emplace_back(node.induction_var());
  }
  void completeLoop(const LoopNode& node) { ivar_stack_.pop_back(); }

  void visitVarLoc(const VarLocNode& node) { inside_write_ = true; }
  void completeVarLoc(const VarLocNode& node) { inside_write_ = false; }

  void visitTensorVarRef(const TensorVarRefNode& node) {
    if (!inside_write_) return;

    size_t axis = ivar_stack_.back().axis();
    bool depends_on_enclosing_loop = false;
    for (const auto& axis_index : node.index_expr().axis_indices()) {
      depends_on_enclosing_loop =
          depends_on_enclosing_loop || axis_index.var.axis() == axis;
    }

    if (!depends_on_enclosing_loop) {
      // Loop iterations write to the same place. We should create a local
      // accum.
      tvars_need_accum_[node]++;
    }
  }

  static std::unordered_map<TensorVarRefNode, int> findTensorWritesNeedAccum(
      FunctionNodePtr ir) {
    AccumWriteFinder visitor;
    visitor.visit(ir);
    return visitor.tvars_need_accum_;
  }

 private:
  // map from each tvar write to the number of local accumulators it needs.
  std::unordered_map<TensorVarRefNode, int> tvars_need_accum_;
  std::vector<InductionVarNode> ivar_stack_;
  bool inside_write_;
};

/*
 * Inserts all local accumulators in the right places, the "right place" for a
 * given tensor write T being the outermost loop that contains all induction
 * variables referenced in T's index expression.
 *
 * Note that this algorithm only works if all induction variables are _globally
 * unique_. If we have two loops with the same named induction variables, then
 * the algorithm will insert all local needed local accumulators into both
 * loops. Thus if we have two separate loop chains with a write in each loop, we
 * will end up with four local accumulators.
 */
class LocalAccumRewriter : public PathCopyVisitor<LocalAccumRewriter> {
 public:
  LocalAccumRewriter(
      const std::unordered_map<TensorVarRefNode, int>& tvars_need_accum)
      : is_parallel_(false), seen_parallel_(false) {
    auto tvar_name = [](const TensorVarRefNode& tvar) {
      std::string tvar_local_name;
      tvar_local_name += tvar.name();
      for (const auto& axis_index : tvar.index_expr().axis_indices()) {
        tvar_local_name +=
            "_" +
            (axis_index.coeff == 1 ? "" : std::to_string(axis_index.coeff));
        tvar_local_name +=
            axis_index.var.name() +
            (axis_index.var.tile_level() == -1
                 ? ""
                 : "T" + std::to_string(axis_index.var.tile_level()));
        tvar_local_name +=
            (axis_index.offset == 0 ? "" : std::to_string(axis_index.offset));
      }

      return tvar_local_name;
    };

    for (const auto& [tvar, count] : tvars_need_accum) {
      std::string tvar_local_name = tvar_name(tvar);
      for (int i = 0; i < count; ++i) {
        tvars_need_accum_[tvar].emplace_back(ScalarVarNode(
            tvar_local_name + "_" + std::to_string(i), Type::kFloat));
      }

      tvar_index_[tvar] = 0;
    }
  }

  void visitFunction(const FunctionNode& node) {
    is_parallel_ = node.is_parallel();
  }

  void visitLoop(const LoopNode& node) {
    enclosing_ivars_.emplace_back(node.induction_var());
    seen_parallel_ = seen_parallel_ || node.is_parallel();
  }

  PtrResult<VarExprNode> completeVarExpr(
      std::shared_ptr<VarExprNode> node,
      UniquePtrResult<VarRefNode>&& var_ref_result) {
    if (node->var_ref().kind() != IrNode::Kind::kTensorVarRef) {
      return PathCopyVisitor<LocalAccumRewriter>::completeVarExpr(
          node, std::move(var_ref_result));
    }

    const TensorVarRefNode& tvar =
        static_cast<const TensorVarRefNode&>(node->var_ref());
    if (tvar_index_.find(tvar) == tvar_index_.end()) {
      return PathCopyVisitor<LocalAccumRewriter>::completeVarExpr(
          node, std::move(var_ref_result));
    }

    // increment index at outer asgn node.
    int index = tvar_index_[tvar];
    const ScalarVarNode& local_accum = tvars_need_accum_[tvar][index];
    return PtrResult<VarExprNode>(
        true, VarExprNode::create(DefaultVarRefNode(local_accum)));
  }

  PtrResult<LoopNode> completeLoop(
      std::shared_ptr<LoopNode> node,
      UniquePtrResult<InductionVarNode>&& induction_var_result,
      PtrResult<StmtNode> body_result) {
    enclosing_ivars_.pop_back();

    // Check if inserting accumulator here will cause accumulator to not be
    // thread-local.
    bool is_non_thread_local =
        is_parallel_ && !seen_parallel_ && !node->is_parallel();
    if (is_non_thread_local) {
      return PathCopyVisitor<LocalAccumRewriter>::completeLoop(
          node, std::move(induction_var_result), body_result);
    }

    // Check if we should insert any local accumulators here.
    std::vector<TensorVarRefNode> tvars_need_accum_here;
    for (const auto& [tvar, local_accum_names] : tvars_need_accum_) {
      bool loop_contains_index_expr = false;
      bool all_ivars_included = true;
      for (const auto& axis_index : tvar.index_expr().axis_indices()) {
        if (node->induction_var() == axis_index.var) {
          loop_contains_index_expr = true;
          continue;
        }

        bool found_ivar = false;
        for (const InductionVarNode& ivar : enclosing_ivars_) {
          if (ivar == axis_index.var) {
            found_ivar = true;
            break;
          }
        }

        if (!found_ivar) {
          all_ivars_included = false;
          break;
        }
      }

      if (loop_contains_index_expr && all_ivars_included) {
        tvars_need_accum_here.emplace_back(tvar);
      }
    }

    // No accumulators. Proceed as normal.
    if (tvars_need_accum_here.empty()) {
      return PathCopyVisitor<LocalAccumRewriter>::completeLoop(
          node, std::move(induction_var_result), body_result);
    }

    // Accumulators. Create inner body first (loop + writes)
    std::vector<StmtNodePtr> inner_stmts = {body_result.node};
    for (const TensorVarRefNode& tvar : tvars_need_accum_here) {
      for (const ScalarVarNode& local_accum : tvars_need_accum_[tvar]) {
        inner_stmts.emplace_back(AsgnNode::create(
            VarLocNode(tvar),
            VarExprNode::create(DefaultVarRefNode(local_accum))));
      }
    }

    // Then create surrounding lets.
    StmtNodePtr new_body = SeqNode::create(inner_stmts);
    for (const TensorVarRefNode& tvar : tvars_need_accum_here) {
      const std::vector<ScalarVarNode>& local_accums = tvars_need_accum_[tvar];
      for (int i = local_accums.size() - 1; i >= 0; --i) {
        const ScalarVarNode& local_accum = local_accums[i];
        new_body =
            LetNode::create(VarDeclNode(local_accum),
                            ConstNode::create(Type::kFloat, 0), new_body);
      }
    }

    return PtrResult<LoopNode>(
        true, LoopNode::create(*(induction_var_result.node),
                               node->lower_bound(), node->upper_bound(),
                               node->stride(), new_body, node->is_parallel()));
  }

  PtrResult<AsgnNode> completeAsgn(std::shared_ptr<AsgnNode> node,
                                   Result<VarLocNode> var_loc_result,
                                   PtrResult<ExprNode> expr_result) {
    if (node->var_loc().var_ref().kind() != IrNode::Kind::kTensorVarRef) {
      return PathCopyVisitor<LocalAccumRewriter>::completeAsgn(
          node, var_loc_result, expr_result);
    }

    const TensorVarRefNode& tvar =
        static_cast<const TensorVarRefNode&>(node->var_loc().var_ref());
    if (tvar_index_.find(tvar) == tvar_index_.end()) {
      return PathCopyVisitor<LocalAccumRewriter>::completeAsgn(
          node, var_loc_result, expr_result);
    }

    int index = tvar_index_[tvar]++;
    const ScalarVarNode& local_accum = tvars_need_accum_[tvar][index];
    return PtrResult<AsgnNode>(
        true, AsgnNode::create(VarLocNode(DefaultVarRefNode(local_accum)),
                               expr_result.node));
  }

  static FunctionNodePtr rewrite(
      FunctionNodePtr ir,
      const std::unordered_map<TensorVarRefNode, int>& tvars_need_accum) {
    LocalAccumRewriter rewriter(tvars_need_accum);
    PtrResult<FunctionNode> result = rewriter.visit(ir);
    return result.node;
  }

 private:
  // need these booleans to prevent inserting non thread-local accumulators.
  bool is_parallel_;
  bool seen_parallel_;
  std::unordered_map<TensorVarRefNode, int> tvar_index_;

  // Map from tensor write to local accumulator name.
  std::unordered_map<TensorVarRefNode, std::vector<ScalarVarNode>>
      tvars_need_accum_;
  std::vector<InductionVarNode> enclosing_ivars_;
};

}  // namespace

/* static */ FunctionNodePtr LocalAccumTransform::apply(FunctionNodePtr ir) {
  std::unordered_map<TensorVarRefNode, int> tvars_need_accum =
      AccumWriteFinder::findTensorWritesNeedAccum(ir);
  if (tvars_need_accum.empty()) {
    return ir;
  }

  return LocalAccumRewriter::rewrite(ir, tvars_need_accum);
}
}  // namespace peachyir
