#pragma once

#include "peachy_ir_agent/ir.h"

namespace peachyir {

/*
 * Finds all variables that need synchronization on write.
 */
class LocalAccumTransform {
 public:
  static constexpr char kName[] = "local_accum";
  static FunctionNodePtr apply(FunctionNodePtr ir);
};

}  // namespace peachyir
