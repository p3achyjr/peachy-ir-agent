#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "peachy_ir_agent/ir.h"

namespace peachyir {
struct TransformResult {
  TransformResult(bool satisfies) : satisfies(satisfies) {}
  TransformResult(std::string error_msg)
      : satisfies(false), error_msg(error_msg) {}
  explicit TransformResult(FunctionNodePtr ir) : satisfies(true), ir(ir) {}

  bool satisfies;
  std::string error_msg;
  FunctionNodePtr ir;

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
