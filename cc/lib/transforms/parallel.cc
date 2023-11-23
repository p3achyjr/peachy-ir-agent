#include "peachy_ir_agent/transforms/parallel.h"

#include <algorithm>

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/visitor/path_copy_visitor.h"

namespace peachyir {
namespace {

class ParallelRewriter : public PathCopyVisitor<ParallelRewriter> {
 public:
  using PathCopyVisitor<ParallelRewriter>::visit;
  ParallelRewriter(FunctionNodePtr ir, const LoopNode& loop_node)
      : loop_node_(loop_node) {}

  PtrResult<LoopNode> visit(std::shared_ptr<LoopNode> node) {
    if (node->induction_var() != loop_node_.induction_var()) {
      return PathCopyVisitor<ParallelRewriter>::visit(node);
    }

    return PtrResult<LoopNode>(
        true, LoopNode::create(node->induction_var(), node->lower_bound(),
                               node->upper_bound(), node->stride(),
                               node->body(), true /* is_parallel */));
  }

  PtrResult<FunctionNode> completeFunction(
      std::shared_ptr<FunctionNode> node,
      std::vector<Result<VarDeclNode>> arg_results,
      std::vector<Result<DefineNode>> define_results,
      PtrResult<StmtNode> body_result) {
    return PtrResult<FunctionNode>(
        true,
        FunctionNode::create(node->name(), extract(arg_results),
                             extract(define_results), node->axes_info(),
                             extract(body_result), true /* is_parallel */));
  }

  static FunctionNodePtr rewrite(FunctionNodePtr ir,
                                 const LoopNode& loop_node) {
    ParallelRewriter rewriter(ir, loop_node);
    FunctionNodePtr new_ir = rewriter.visit(ir).node;
    return new_ir;
  }

 private:
  const LoopNode& loop_node_;
};

TransformResult canApply(FunctionNodePtr ir, const IrNode& node) {
  if (node.kind() != IrNode::Kind::kLoop) {
    return errorMsg("`%s` Expected `kLoop`, got `%s`", ParallelTransform::kName,
                    str(node.kind()).c_str());
  }

  if (ir->is_parallel()) {
    return errorMsg("`%s` This function is already parallelized.",
                    ParallelTransform::kName);
  }

  return true;
}

}  // namespace

/* static */ TransformResult ParallelTransform::apply(FunctionNodePtr ir,
                                                      const IrNode& node) {
  TransformResult rewrite_result = canApply(ir, node);
  if (!rewrite_result) {
    return rewrite_result;
  }

  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  rewrite_result.ir = ParallelRewriter::rewrite(ir, loop_node);
  return rewrite_result;
}
}  // namespace peachyir
