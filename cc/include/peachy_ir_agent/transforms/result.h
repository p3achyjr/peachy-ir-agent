#pragma once

#include <sstream>
#include <string>

namespace peachyir {
struct Result {
  Result(bool satisfies) : satisfies(satisfies) {}
  Result(bool satisfies, std::string error_msg)
      : satisfies(satisfies), error_msg(error_msg) {}
  bool satisfies;
  std::string error_msg;

  operator bool() { return satisfies; }
};

template <typename... Args>
std::string errorMsg(const char* format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format, args...) + 1;
  if (size_s <= 0) return "";

  char buf[static_cast<size_t>(size_s)];
  snprintf(buf, static_cast<size_t>(size_s), format, args...);
  return std::string(buf);
}
}  // namespace peachyir
