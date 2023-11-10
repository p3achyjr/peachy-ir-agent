#include "matmul.h"

#include "peachy_ir_agent/ir.h"

namespace peachyir {
namespace kernels {

FunctionNode matmulIr() {
  // def matmul(C: (M, N), A: (M, K), B: (K, N)):
  //   loop "i" (0, M) 1 in
  //     loop "j" (0, N) 1 in
  //       loop "k" (0, K) 1 in
  //         C[i, j] = C[i, j] + A[i, k] * B[k, j]
  return FunctionNode({}, {}, NopNode());
}

}  // namespace kernels
}  // namespace peachyir
