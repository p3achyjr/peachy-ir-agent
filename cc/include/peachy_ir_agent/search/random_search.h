#pragma once

#include "peachy_ir_agent/search/cost_model.h"
#include "peachy_ir_agent/search/search.h"

namespace peachyir {

/*
 * Performs `search_budget` iterations of random search.
 *
 * First picks a number `s` between 5 and 30. Then uniformly applies transforms
 * until reaching the budget, or until there are no more valid moves.
 */
class RandomSearch : public Search {
 public:
  RandomSearch();
  RandomSearch(size_t search_budget);
  Result search(FunctionNodePtr initial_ir, CostModel* cost_model) override;

 private:
  size_t search_budget_;
};

}  // namespace peachyir
