#pragma once

#include "peachy_ir_agent/ir.h"

namespace peachyir {
class CostModel;

class Search {
 public:
  struct Result {
    FunctionNodePtr ir;
    float cost;
  };

  virtual ~Search() = default;
  virtual Result search(FunctionNodePtr initial_ir, CostModel* cost_model) = 0;
};

}  // namespace peachyir
