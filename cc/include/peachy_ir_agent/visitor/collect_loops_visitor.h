#pragma once

#include <iostream>

#include "peachy_ir_agent/ir_visitor.h"

namespace peachyir {

class CollectLoopsVisitor : public IrVisitor<CollectLoopsVisitor> {
 public:
  void visitLoop(const LoopNode& node) { loop_nodes_.emplace_back(node); }

  static std::vector<LoopNode> collect(FunctionNodePtr ir) {
    CollectLoopsVisitor visitor;
    visitor.visit(ir);
    return visitor.loop_nodes_;
  }

 private:
  std::vector<LoopNode> loop_nodes_;
};

}  // namespace peachyir
