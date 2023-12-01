#include "peachy_ir_agent/search/random_search.h"

#include "peachy_ir_agent/codegen/codegen.h"
#include "peachy_ir_agent/rand.h"
#include "peachy_ir_agent/search/collect_transforms.h"
#include "peachy_ir_agent/visitor/collect_loops_visitor.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

RandomSearch::RandomSearch() : search_budget_(10000) {}
RandomSearch::RandomSearch(size_t search_budget)
    : search_budget_(search_budget) {}

Search::Result RandomSearch::search(FunctionNodePtr initial_ir,
                                    CostModel* cost_model) {
  static constexpr size_t kLogInterval = 100;

  float min_cost = -1;
  FunctionNodePtr min_ir;

  PRng prng;
  for (int i = 0; i < search_budget_; ++i) {
    int num_turns = randInt(prng, 0, 15);
    FunctionNodePtr ir = initial_ir;
    for (int turn = 0; turn < num_turns; ++turn) {
      // find all actions from all loops.
      std::vector<LoopNode> loop_nodes = CollectLoopsVisitor::collect(ir);
      std::vector<FunctionNodePtr> next_irs = collectTransforms(ir, loop_nodes);
      if (next_irs.empty()) {
        break;
      }

      int ir_index = randInt(prng, 0, next_irs.size());
      ir = next_irs[ir_index];
    }

    float cost = cost_model->cost(ir);
    if (min_cost == -1 || cost < min_cost) {
      min_cost = cost;
      min_ir = ir;
      std::cout << "Best Program:\n"
                << IrPrinter::print(min_ir) << "\nTime: " << min_cost << "\n";
    }
  }

  return Result{};
}

}  // namespace peachyir
