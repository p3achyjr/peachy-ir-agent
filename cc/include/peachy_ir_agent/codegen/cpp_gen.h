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

/*
 * Generates Allocation Code from IR.
 */
class CppAllocGen {
 public:
  static constexpr char kName[] = "cpp_alloc_gen";
  static std::string apply(FunctionNodePtr ir);
};

/*
 * Generates Invocation Code from IR.
 */
class CppInvokeGen {
 public:
  static constexpr char kName[] = "cpp_invoke_gen";
  static std::string apply(FunctionNodePtr ir);
};

/*
 * Generates Initialization Code from IR.
 */
class CppInitGen {
 public:
  static constexpr char kName[] = "cpp_init_gen";
  static std::string apply(FunctionNodePtr ir);
};

/*
 * Generates Reset Code from IR.
 */
class CppResetGen {
 public:
  static constexpr char kName[] = "cpp_reset_gen";
  static std::string apply(FunctionNodePtr ir);
};

}  // namespace peachyir
