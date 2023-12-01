#pragma once

#include <string>

#include "peachy_ir_agent/ir.h"

namespace peachyir {

/*
 * Takes an IR, and generates C++ code from it.
 */
std::string codegen(FunctionNodePtr ir);

}
