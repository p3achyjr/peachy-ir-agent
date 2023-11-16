#pragma once

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/ir_visitor.h"
#include "util.h"

namespace peachyir {

/*
 * Visitor that finds the innermost loop that encloses `loop` containing the
 * same induction variable.
 */
class EnclosingLoopStrideFinder : public IrVisitor<EnclosingLoopStrideFinder> {
 public:
  using IrVisitor<EnclosingLoopStrideFinder>::visit;
  EnclosingLoopStrideFinder(FunctionNodePtr ir, const LoopNode& loop_node);

  void visit(const LoopNode& node);
  void completeLoop(const LoopNode& node);
  static size_t eval(FunctionNodePtr ir, const LoopNode& node);

 private:
  FunctionNodePtr ir_;
  const LoopNode& loop_node_;
  size_t enclosing_loop_stride_;
  bool found_loop_;
  bool found_enclosing_loop_;
};
}  // namespace peachyir
