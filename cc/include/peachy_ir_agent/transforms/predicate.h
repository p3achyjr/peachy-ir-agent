#pragma once

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/result.h"

namespace peachyir {

class Predicate {
 public:
  Predicate() = delete;
  Result satisfies(const IrNode& node);
};

}  // namespace peachyir
