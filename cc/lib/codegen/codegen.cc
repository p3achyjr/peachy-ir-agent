#include "peachy_ir_agent/codegen/codegen.h"

#include "peachy_ir_agent/codegen/cpp_gen.h"
#include "peachy_ir_agent/codegen/critical_section.h"
#include "peachy_ir_agent/codegen/local_accum.h"

namespace peachyir {

std::string codegen(FunctionNodePtr ir) {
  ir = LocalAccumTransform::apply(ir);
  ir = LocalAccumTransform::apply(ir);

  return CppGen::apply(ir);
}

}  // namespace peachyir
