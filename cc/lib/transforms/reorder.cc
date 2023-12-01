#include "peachy_ir_agent/transforms/reorder.h"

#include "peachy_ir_agent/visitor/path_copy_visitor.h"

namespace peachyir {
namespace {

class ReorderRewriter : public PathCopyVisitor<ReorderRewriter> {
 public:
  using PathCopyVisitor<ReorderRewriter>::visit;
  ReorderRewriter(FunctionNodePtr ir, const LoopNode& loop_node)
      : loop_node_(loop_node) {}

  PtrResult<LoopNode> visit(std::shared_ptr<LoopNode> node) {
    if (node->induction_var() != loop_node_.induction_var()) {
      return PathCopyVisitor<ReorderRewriter>::visit(node);
    }

    LoopNodePtr inner_loop = std::static_pointer_cast<LoopNode>(node->body());
    StmtNodePtr inner_body = inner_loop->body();
    LoopNodePtr new_inner_loop = LoopNode::create(
        node->induction_var(), node->lower_bound(), node->upper_bound(),
        node->stride(), inner_body, node->is_parallel());
    return PtrResult<LoopNode>(
        true,
        LoopNode::create(inner_loop->induction_var(), inner_loop->lower_bound(),
                         inner_loop->upper_bound(), inner_loop->stride(),
                         new_inner_loop, inner_loop->is_parallel()));
  }

  static FunctionNodePtr rewrite(FunctionNodePtr ir,
                                 const LoopNode& loop_node) {
    ReorderRewriter rewriter(ir, loop_node);
    FunctionNodePtr new_ir = rewriter.visit(ir).node;
    return new_ir;
  }

 private:
  const LoopNode& loop_node_;
};

TransformResult canApply(FunctionNodePtr ir, const IrNode& node) {
  if (node.kind() != IrNode::Kind::kLoop) {
    return errorMsg("`%s` Expected `kLoop`, got `%s`", ReorderTransform::kName,
                    str(node.kind()).c_str());
  }

  const LoopNode& outer_loop = static_cast<const LoopNode&>(node);
  if (outer_loop.body()->kind() != IrNode::Kind::kLoop) {
    return errorMsg("`%s` Expected loop body to be `kLoop`, got `%s`",
                    ReorderTransform::kName,
                    str(outer_loop.body()->kind()).c_str());
  }

  const LoopNode& inner_loop = static_cast<const LoopNode&>(
      *std::static_pointer_cast<LoopNode>(outer_loop.body()));
  if (outer_loop.induction_var().axis() == inner_loop.induction_var().axis()) {
    return errorMsg("`%s` Loops iterate along the same axis. Cannot reorder.",
                    ReorderTransform::kName);
  }

  return true;
}

}  // namespace

/* static */ TransformResult ReorderTransform::apply(FunctionNodePtr ir,
                                                     const IrNode& node) {
  TransformResult rewrite_result = canApply(ir, node);
  if (!rewrite_result) {
    return rewrite_result;
  }

  // This loop is either a LoopNode or ParLoopNode, but they have the same
  // fields.
  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  rewrite_result.ir = ReorderRewriter::rewrite(ir, loop_node);
  return rewrite_result;
}

}  // namespace peachyir
