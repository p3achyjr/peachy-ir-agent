#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/predicate.h"

namespace peachyir {

class Transform {
 public:
  virtual ~Transform() = default;

  virtual bool canApply(const IrNode& node) = 0;
  virtual FunctionNodePtr apply(FunctionNodePtr ir, const IrNode& node) = 0;
};

}  // namespace peachyir
