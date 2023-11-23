#include "peachy_ir_agent/transforms/tile_resize.h"

#include <algorithm>

#include "peachy_ir_agent/base.h"
#include "tile_common.h"
#include "util.h"

namespace peachyir {
namespace {

TransformResult canApply(FunctionNodePtr ir, const IrNode& node) {
  if (node.kind() != IrNode::Kind::kLoop) {
    return errorMsg("`%s` Expected `kLoop`, got `%s`",
                    TileResizeTransform::kName, str(node.kind()).c_str());
  }

  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  const size_t loop_stride = eval_stride(ir, loop_node);
  if (loop_stride == 1) {
    return errorMsg(
        "`%s` The current loop is not tiled. Tile first before changing tile "
        "size.",
        TileResizeTransform::kName);
  }

  const size_t loop_axis = loop_node.induction_var().axis();
  const size_t axis_length = ir->axis_len(loop_axis);
  const size_t proposed_tile_size = loop_stride * 2;
  ASSERT(proposed_tile_size > 0 && proposed_tile_size % 4 == 0);

  return checkCommonTileConditions(ir, loop_node, proposed_tile_size,
                                   TileResizeTransform::kName);
}

}  // namespace

/* static */ TransformResult TileResizeTransform::apply(FunctionNodePtr ir,
                                                        const IrNode& node) {
  TransformResult rewrite_result = canApply(ir, node);
  if (!rewrite_result) {
    return rewrite_result;
  }

  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  const std::string& define_name = std::get<std::string>(loop_node.stride());
  const size_t loop_stride = eval_stride(ir, loop_node);
  const size_t proposed_tile_size = loop_stride * 2;

  std::vector<DefineNode> new_defines;
  std::transform(ir->defines().begin(), ir->defines().end(),
                 std::back_inserter(new_defines),
                 [&define_name, proposed_tile_size](const DefineNode& define) {
                   return define.name() == define_name
                              ? DefineNode(define_name, proposed_tile_size)
                              : define;
                 });

  return TransformResult(FunctionNode::create(
      ir->name(), ir->args(), new_defines, ir->axes_info(), ir->body()));
}
}  // namespace peachyir
