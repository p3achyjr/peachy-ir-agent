#include "peachy_ir_agent/rand.h"

#include <chrono>

#ifndef PCG_MULT
#define PCG_MULT 6364136223846793005u
#endif

#ifndef PCG_ADD
#define PCG_ADD 1442695040888963407u
#endif

namespace peachyir {
namespace {

struct RandResult {
  uint64_t new_state;
  uint32_t rand;
};

inline uint32_t rotate32(uint32_t x, unsigned r) {
  return x >> r | x << (-r & 31);
}

// use top 5 bits to determine rotation (log2(32) = 5)
inline RandResult pcg32(uint64_t state) {
  uint64_t x = state;
  unsigned count = (unsigned)(x >> 59);  // 59 = 64 - 5

  state = x * PCG_MULT + PCG_ADD;
  x ^= x >> 18;  // 18 = (64 - 27)/2
  return RandResult{
      state, rotate32((uint32_t)(x >> 27), count)  // 27 = 32 - 5
  };
}

}  // namespace

PRng::PRng()
    : PRng(std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
               .count()) {}

PRng::PRng(uint64_t seed) { state_ = seed + PCG_ADD; }

uint32_t PRng::next() {
  RandResult rand_result = pcg32(state_);
  state_ = rand_result.new_state;

  return rand_result.rand;
}
}  // namespace peachyir
