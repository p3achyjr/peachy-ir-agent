#include "util.h"

namespace peachyir {
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
}  // namespace peachyir
