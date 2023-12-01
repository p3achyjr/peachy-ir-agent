#pragma once

#include "peachy_ir_agent/search/cost_model.h"

namespace peachyir {

class CpuCostModel : public CostModel {
 public:
  ~CpuCostModel() override = default;
  float cost(FunctionNodePtr ir) override;
};

}  // namespace peachyir
