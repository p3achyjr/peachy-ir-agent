#include "peachy_ir_agent/transforms/tile.h"

#include "peachy_ir_agent/base.h"
#include "peachy_ir_agent/ir.h"
#include "peachy_ir_agent/ir_visitor.h"
#include "peachy_ir_agent/visitor/path_copy_visitor.h"

namespace peachyir {
namespace {
size_t eval_stride(FunctionNodePtr ir, const LoopNode& loop_node) {
  return std::visit(
      [ir](auto&& b) -> size_t {
        using T = std::decay_t<decltype(b)>;
        if constexpr (std::is_same_v<T, std::string>) {
          for (const DefineNode& define : ir->defines()) {
            if (define.name() == b) {
              return define.val();
            }
          }

          return 0;
        } else {
          return b;
        }
      },
      loop_node.stride());
}

/*
 * Visitor that finds the innermost loop that encloses `loop` containing the
 * same induction variable.
 */
class EnclosingLoopStrideFinder : public IrVisitor<EnclosingLoopStrideFinder> {
 public:
  using IrVisitor<EnclosingLoopStrideFinder>::visit;
  EnclosingLoopStrideFinder(FunctionNodePtr ir, const LoopNode& loop_node)
      : ir_(ir),
        loop_node_(loop_node),
        enclosing_loop_stride_(0),
        found_loop_(false),
        found_enclosing_loop_(false) {}

  void visit(const LoopNode& node) {
    if (node.induction_var() == loop_node_.induction_var()) {
      // This is the loop we are interested in. Mark state and return early.
      found_loop_ = true;
      return;
    }

    // Otherwise, continue traversing IR.
    IrVisitor<EnclosingLoopStrideFinder>::visit(node);
  }

  void completeLoop(const LoopNode& node) {
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

  static size_t eval(FunctionNodePtr ir, const LoopNode& node) {
    EnclosingLoopStrideFinder visitor(ir, node);
    visitor.visit(ir);

    return visitor.enclosing_loop_stride_;
  }

 private:
  FunctionNodePtr ir_;
  const LoopNode& loop_node_;
  size_t enclosing_loop_stride_;
  bool found_loop_;
  bool found_enclosing_loop_;
};

/*
 * Visitor that applies the tiling transform.
 */
class TileRewriter : public PathCopyVisitor<TileRewriter> {
 public:
  using PathCopyVisitor<TileRewriter>::visit;
  TileRewriter(FunctionNodePtr ir, const LoopNode& loop_node, size_t tile_size)
      : loop_node_(loop_node),
        define_name_("D" + std::to_string(ir->defines().size())),
        axis_(loop_node.induction_var().axis()),
        next_tile_(ir->axis_tiles(axis_)),
        tile_size_(tile_size) {}

  PtrResult<FunctionNode> completeFunction(
      FunctionNodePtr node, std::vector<Result<VarDeclNode>> arg_results,
      std::vector<Result<DefineNode>> define_results,
      PtrResult<StmtNode> body_result) {
    std::vector<DefineNode> defines = extract(define_results);
    defines.emplace_back(DefineNode(define_name_, tile_size_));

    std::vector<FunctionNode::AxisInfo> axes_info = node->axes_info();
    axes_info[axis_].tiles++;
    return PtrResult<FunctionNode>(
        true, FunctionNode::create(node->name(), extract(arg_results), defines,
                                   axes_info, extract(body_result)));
  }

  PtrResult<LoopNode> completeLoop(
      LoopNodePtr node, UniquePtrResult<InductionVarNode> induction_var_result,
      PtrResult<StmtNode> body_result) {
    bool did_change = changed(induction_var_result.changed, body_result);
    if (did_change) {
      // It's impossible for two loops to get tiled.
      return PtrResult<LoopNode>{
          true, LoopNode::create(*(induction_var_result.node),
                                 node->lower_bound(), node->upper_bound(),
                                 node->stride(), extract(body_result))};
    } else if (node->induction_var() != loop_node_.induction_var()) {
      // This is not the loop we want to tile.
      return PtrResult<LoopNode>{false, node};
    }

    // This is the loop we want to tile.
    const size_t axis = node->induction_var().axis();
    const std::string name = node->induction_var().name();

    InductionVarNode outer_ivar(name, axis, next_tile_);
    LoopNode::Bound outer_lb = node->lower_bound();
    LoopNode::Bound outer_ub = node->upper_bound();
    LoopNode::Stride outer_stride = define_name_;

    InductionVarNode inner_ivar = node->induction_var();
    LoopNode::Bound inner_lb = LoopNode::CompositeBound(outer_ivar);
    LoopNode::Bound inner_ub =
        LoopNode::CompositeBound(1, outer_ivar, define_name_);

    LoopNodePtr inner_loop = LoopNode::create(
        inner_ivar, inner_lb, inner_ub, node->stride(), extract(body_result));
    LoopNodePtr outer_loop = LoopNode::create(outer_ivar, outer_lb, outer_ub,
                                              outer_stride, inner_loop);
    return PtrResult<LoopNode>{true, outer_loop};
  }

  static FunctionNodePtr rewrite(FunctionNodePtr ir, const LoopNode& loop_node,
                                 size_t tile_size) {
    TileRewriter rewriter(ir, loop_node, tile_size);
    FunctionNodePtr new_ir = rewriter.visit(ir).node;
    return new_ir;
  }

 private:
  const LoopNode& loop_node_;
  const std::string define_name_;
  const size_t axis_;
  const size_t next_tile_;
  const size_t tile_size_;
};

}  // namespace

/* static */ Result TileTransform::canApply(FunctionNodePtr ir,
                                            const IrNode& node) {
  if (node.kind() != IrNode::Kind::kLoop) {
    return Result(false,
                  errorMsg("Expected `kLoop`, got `%s`", str(node.kind())));
  }

  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  const size_t loop_axis = loop_node.induction_var().axis();
  const size_t loop_stride = eval_stride(ir, loop_node);
  const size_t axis_length = ir->axis_len(loop_axis);
  const size_t proposed_tile_size = loop_stride == 1 ? 4 : loop_stride * 2;
  ASSERT(proposed_tile_size > 0 && proposed_tile_size % 4 == 0);

  if (proposed_tile_size >= axis_length) {
    return Result(false,
                  errorMsg("Proposed tile size (%zu) >= axis length (%zu).",
                           proposed_tile_size, axis_length));
  }

  if (axis_length % proposed_tile_size != 0) {
    return Result(false,
                  errorMsg("Proposed tile size (%zu) does not divide axis "
                           "length (%zu). This is not yet supported.",
                           proposed_tile_size, axis_length));
  }

  const size_t enclosing_loop_stride =
      EnclosingLoopStrideFinder::eval(ir, loop_node);

  if (enclosing_loop_stride != 0 &&
      enclosing_loop_stride <= proposed_tile_size) {
    // this means that either our tile is too big, or we are introducing a
    // redundant level of tiling.
    return Result(
        false,
        errorMsg("Enclosing tile size (%zu) is <= proposed tile size (%zu).",
                 enclosing_loop_stride, proposed_tile_size));
  }

  return true;
}

/* static */ FunctionNodePtr TileTransform::apply(FunctionNodePtr ir,
                                                  const IrNode& node) {
  // call after checking `canApply`.
  const LoopNode& loop_node = static_cast<const LoopNode&>(node);
  const size_t loop_stride = eval_stride(ir, loop_node);
  const size_t tile_size = loop_stride == 1 ? 4 : loop_stride * 2;

  return TileRewriter::rewrite(ir, loop_node, tile_size);
}

}  // namespace peachyir
