#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/result.h"

namespace peachyir {

/*
 * Tiles a loop on a single axis one level.
 */
class TileTransform {
 public:
  static constexpr char kName[] = "tile";
  static TransformResult apply(FunctionNodePtr ir, const IrNode& node);
};

}  // namespace peachyir
