#pragma once
#include <cassert>
#include <iostream>
#include <sstream>

class LogSink {
 public:
  enum Severity : uint8_t {
    INFO = 0,
    FATAL = 1,
  };

  LogSink(Severity sev) : sev_(sev) {}
  ~LogSink() { std::cerr << ss_.str() << "\n"; }

  template <typename T>
  LogSink& operator<<(T&& x) {
    ss_ << x;
    return *this;
  }

 private:
  std::stringstream ss_;
  Severity sev_;
};

#define LOG(severity) \
  LogSink(LogSink::Severity::severity) << __FILE__ << ":" << __LINE__ << " "

#define ABORT(format, ...)                 \
  do {                                     \
    {                                      \
      char buf[2048];                      \
      sprintf(buf, format, ##__VA_ARGS__); \
      LOG(INFO) << buf;                    \
    }                                      \
    std::abort();                          \
  } while (0);

#define ASSERT assert  // syntatic sugar.
