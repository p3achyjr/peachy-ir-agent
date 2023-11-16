#include "tile_common.h"

#include "enclosing_loop_stride_finder.h"
#include "util.h"

namespace peachyir {

TransformResult checkCommonTileConditions(FunctionNodePtr ir,
                                          const LoopNode& loop_node,
                                          const size_t proposed_tile_size,
                                          std::string transform_name) {
  const size_t loop_axis = loop_node.induction_var().axis();
  const size_t axis_length = ir->axis_len(loop_axis);
  const size_t loop_stride = eval_stride(ir, loop_node);
  if (proposed_tile_size >= axis_length) {
    return errorMsg("`%s` Proposed tile size (%zu) >= axis length (%zu).",
                    transform_name.c_str(), proposed_tile_size, axis_length);
  }

  if (axis_length % proposed_tile_size != 0) {
    return errorMsg(
        "`%s` Proposed tile size (%zu) does not divide axis "
        "length (%zu). This is not yet supported.",
        transform_name.c_str(), proposed_tile_size, axis_length);
  }

  const size_t enclosing_loop_stride =
      EnclosingLoopStrideFinder::eval(ir, loop_node);

  if (enclosing_loop_stride != 0 &&
      enclosing_loop_stride <= proposed_tile_size) {
    // this means that either our tile is too big, or we are introducing a
    // redundant level of tiling.
    return errorMsg(
        "`%s` Enclosing tile size (%zu) is <= proposed tile size (%zu).",
        transform_name.c_str(), enclosing_loop_stride, proposed_tile_size);
  }

  return true;
}

}  // namespace peachyir
