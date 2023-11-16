#include "enclosing_loop_stride_finder.h"

namespace peachyir {

EnclosingLoopStrideFinder::EnclosingLoopStrideFinder(FunctionNodePtr ir,
                                                     const LoopNode& loop_node)
    : ir_(ir),
      loop_node_(loop_node),
      enclosing_loop_stride_(0),
      found_loop_(false),
      found_enclosing_loop_(false) {}

void EnclosingLoopStrideFinder::visit(const LoopNode& node) {
  if (node.induction_var() == loop_node_.induction_var()) {
    // This is the loop we are interested in. Mark state and return early.
    found_loop_ = true;
    return;
  }

  // Otherwise, continue traversing IR.
  IrVisitor<EnclosingLoopStrideFinder>::visit(node);
}

void EnclosingLoopStrideFinder::completeLoop(const LoopNode& node) {
  if (found_enclosing_loop_) {
    return;
  } else if (!found_loop_) {
    return;
  } else if (!(node.induction_var().axis() ==
               loop_node_.induction_var().axis())) {
    return;
  }

  enclosing_loop_stride_ = eval_stride(ir_, node);
  found_enclosing_loop_ = true;
}

/* static */ size_t EnclosingLoopStrideFinder::eval(FunctionNodePtr ir,
                                                    const LoopNode& node) {
  EnclosingLoopStrideFinder visitor(ir, node);
  visitor.visit(ir);

  return visitor.enclosing_loop_stride_;
}

}  // namespace peachyir
