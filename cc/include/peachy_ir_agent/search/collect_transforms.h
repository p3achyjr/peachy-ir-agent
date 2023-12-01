#pragma once

#include <vector>

#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/transforms/result.h"
#include "peachy_ir_agent/transforms/transforms.h"

namespace peachyir {

template <typename Transform>
void applyTransform(std::vector<FunctionNodePtr>& transformed_irs,
                    FunctionNodePtr ir, const std::vector<LoopNode>& loops) {
  for (const LoopNode& loop : loops) {
    TransformResult result = Transform::apply(ir, loop);
    if (!result) {
      continue;
    }

    transformed_irs.emplace_back(result.ir);
  }
}

template <typename... Transforms>
std::vector<FunctionNodePtr> applyTransforms(
    FunctionNodePtr ir, const std::vector<LoopNode>& loops) {
  std::vector<FunctionNodePtr> transformed_irs;
  (applyTransform<Transforms>(transformed_irs, ir, loops), ...);
  return transformed_irs;
}

inline std::vector<FunctionNodePtr> collectTransforms(
    FunctionNodePtr ir, const std::vector<LoopNode>& loops) {
  return applyTransforms<TileTransform, TileResizeTransform, ReorderTransform>(
      ir, loops);
}
}  // namespace peachyir
