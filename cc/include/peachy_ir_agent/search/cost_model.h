#pragma once

#include "peachy_ir_agent/ir.h"

namespace peachyir {

class CostModel {
 public:
  virtual ~CostModel() = default;
  virtual float cost(FunctionNodePtr ir) = 0;
};

}  // namespace peachyir
