#pragma once

#include "peachy_ir_agent/ir.h"

namespace peachyir {

/*
 * Finds all variables that need synchronization on write.
 */
class CriticalSectionTransform {
 public:
  static constexpr char kName[] = "critical_section";
  static FunctionNodePtr apply(FunctionNodePtr ir);
};

}  // namespace peachyir
