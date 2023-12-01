#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/search/cpu_cost_model.h"
#include "peachy_ir_agent/search/random_search.h"

int main(int argc, char** argv) {
  peachyir::FunctionNodePtr matmul_ir =
      peachyir::kernels::matmulIr(1024, 1024, 1024, "matmul");
  peachyir::CpuCostModel cost_model;
  peachyir::RandomSearch rand_search(100);
  rand_search.search(matmul_ir, &cost_model);
  return 0;
}
