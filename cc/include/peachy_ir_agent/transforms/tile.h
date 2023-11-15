#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/predicate.h"

namespace peachyir {

class TileTransform {
 public:
  static Result canApply(FunctionNodePtr ir, const IrNode& node);
  static FunctionNodePtr apply(FunctionNodePtr ir, const IrNode& node);
};

}  // namespace peachyir
