#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/result.h"

namespace peachyir {

/*
 * Parallelizes the current loop.
 */
class ParallelTransform {
 public:
  static constexpr char kName[] = "parallel";
  static TransformResult apply(FunctionNodePtr ir, const IrNode& node);
};

}  // namespace peachyir
