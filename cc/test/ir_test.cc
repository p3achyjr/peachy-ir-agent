#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include "peachy_ir_agent/ir.h"

#include "doctest/doctest.h"
#include "peachy_ir_agent/kernels/matmul.h"
#include "peachy_ir_agent/visitor/ir_printer.h"

namespace peachyir {

TEST_CASE("Test Matmul") {
  static constexpr char kExpectedStr[] =
      "def matmul (C: (1024, 1024), A: (1024, 512), B: (512, 1024)):\n"
      "  loop i (0, 1024) 1:\n"
      "    loop j (0, 1024) 1:\n"
      "      loop k (0, 512) 1:\n"
      "        C[i, j] = C[i, j] + A[i, k] * B[k, j];\n";
  FunctionNodePtr matmul_ir = kernels::matmulIr(1024, 1024, 512, "matmul");
  std::cerr << IrPrinter::print(matmul_ir);
}
}  // namespace peachyir
