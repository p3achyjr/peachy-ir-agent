#pragma once

#include "peachy_ir_agent/ir.h"

namespace peachyir {

/*
 * Generates C++ code from IR.
 */
class CppGen {
 public:
  static constexpr char kName[] = "cpp_gen";
  static std::string apply(FunctionNodePtr ir);
};

}  // namespace peachyir
