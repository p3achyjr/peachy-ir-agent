#pragma once

#include "peachy_ir_agent/ir.h"

namespace peachyir {
size_t eval_stride(FunctionNodePtr ir, const LoopNode& loop_node);
}  // namespace peachyir
