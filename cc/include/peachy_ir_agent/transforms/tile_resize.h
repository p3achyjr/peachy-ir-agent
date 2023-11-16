#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/result.h"

namespace peachyir {

/*
 * Multiples a single tile size by 2.
 */
class TileResizeTransform {
 public:
  static constexpr char kName[] = "tile_resize";
  static TransformResult apply(FunctionNodePtr ir, const IrNode& node);
};

}  // namespace peachyir
