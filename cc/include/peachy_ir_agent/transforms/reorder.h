#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/result.h"

namespace peachyir {

/*
 * Reorders the loop at `node` and the loop directly underneath.
 */
class ReorderTransform {
 public:
  static constexpr char kName[] = "reorder";
  static TransformResult apply(FunctionNodePtr ir, const IrNode& node);
};

}  // namespace peachyir
