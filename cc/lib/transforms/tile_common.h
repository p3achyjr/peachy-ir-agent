#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/result.h"

namespace peachyir {

/*
 * Performs common tile transform checks.
 */
TransformResult checkCommonTileConditions(FunctionNodePtr ir,
                                          const LoopNode& loop_node,
                                          const size_t proposed_tile_size,
                                          std::string transform_name);

}  // namespace peachyir
