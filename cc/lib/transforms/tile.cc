#include "peachy_ir_agent/transforms/tile.h"

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/visitor/path_copy_visitor.h"
#include "tile_common.h"
#include "util.h"

namespace peachyir {
namespace {

/*
 * Visitor that applies the tiling transform.
 */
class TileRewriter : public PathCopyVisitor<TileRewriter> {
 public:
  using PathCopyVisitor<TileRewriter>::visit;
  TileRewriter(FunctionNodePtr ir, const LoopNode& loop_node, size_t tile_size)
      : loop_node_(loop_node),
        define_name_("D" + std::to_string(ir->defines().size())),
        axis_(loop_node.induction_var().axis()),
        next_tile_(ir->axis_tiles(axis_)),
        tile_size_(tile_size) {}

  PtrResult<FunctionNode> completeFunction(
      FunctionNodePtr node, std::vector<Result<VarDeclNode>> arg_results,
      std::vector<Result<DefineNode>> define_results,
      PtrResult<StmtNode> body_result) {
    std::vector<DefineNode> defines = extract(define_results);
    defines.emplace_back(DefineNode(define_name_, tile_size_));

    std::vector<FunctionNode::AxisInfo> axes_info = node->axes_info();
    axes_info[axis_].tiles++;
    return PtrResult<FunctionNode>(
        true, FunctionNode::create(node->name(), extract(arg_results), defines,
                                   axes_info, extract(body_result)));
  }

  PtrResult<LoopNode> completeLoop(
      LoopNodePtr node, UniquePtrResult<InductionVarNode> induction_var_result,
      PtrResult<StmtNode> body_result) {
    bool did_change = changed(induction_var_result.changed, body_result);
    if (did_change) {
      // It's impossible for two loops to get tiled.
      return PtrResult<LoopNode>{
          true, LoopNode::create(*(induction_var_result.node),
                                 node->lower_bound(), node->upper_bound(),
                                 node->stride(), extract(body_result))};
    } else if (node->induction_var() != loop_node_.induction_var()) {
      // This is not the loop we want to tile.
      return PtrResult<LoopNode>{false, node};
    }

    // This is the loop we want to tile.
    const size_t axis = node->induction_var().axis();
    const std::string name = node->induction_var().name();

    InductionVarNode outer_ivar(name, axis, next_tile_);
    LoopNode::Bound outer_lb = node->lower_bound();
    LoopNode::Bound outer_ub = node->upper_bound();
    LoopNode::Stride outer_stride = define_name_;

    InductionVarNode inner_ivar = node->induction_var();
    LoopNode::Bound inner_lb = LoopNode::CompositeBound(outer_ivar);
    LoopNode::Bound inner_ub =
        LoopNode::CompositeBound(1, outer_ivar, define_name_);

    LoopNodePtr inner_loop = LoopNode::create(
        inner_ivar, inner_lb, inner_ub, node->stride(), extract(body_result));
    LoopNodePtr outer_loop = LoopNode::create(outer_ivar, outer_lb, outer_ub,
                                              outer_stride, inner_loop);
    return PtrResult<LoopNode>{true, outer_loop};
  }

  static FunctionNodePtr rewrite(FunctionNodePtr ir, const LoopNode& loop_node,
                                 size_t tile_size) {
    TileRewriter rewriter(ir, loop_node, tile_size);
    FunctionNodePtr new_ir = rewriter.visit(ir).node;
    return new_ir;
  }

 private:
  const LoopNode& loop_node_;
  const std::string define_name_;
  const size_t axis_;
  const size_t next_tile_;
  const size_t tile_size_;
};

TransformResult canApply(FunctionNodePtr ir, const IrNode& node) {
  if (node.kind() != IrNode::Kind::kLoop) {
    return errorMsg("`%s` Expected `kLoop`, got `%s`", TileTransform::kName,
                    str(node.kind()).c_str());
  }

  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  const size_t loop_stride = eval_stride(ir, loop_node);
  const size_t proposed_tile_size = loop_stride == 1 ? 4 : loop_stride * 2;
  ASSERT(proposed_tile_size > 0 && proposed_tile_size % 4 == 0);

  return checkCommonTileConditions(ir, loop_node, proposed_tile_size,
                                   TileTransform::kName);
}
}  // namespace

/* static */ TransformResult TileTransform::apply(FunctionNodePtr ir,
                                                  const IrNode& node) {
  TransformResult rewrite_result = canApply(ir, node);
  if (!rewrite_result) {
    return rewrite_result;
  }

  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  const size_t loop_stride = eval_stride(ir, loop_node);
  const size_t tile_size = loop_stride == 1 ? 4 : loop_stride * 2;

  rewrite_result.ir = TileRewriter::rewrite(ir, loop_node, tile_size);
  return rewrite_result;
}

}  // namespace peachyir
