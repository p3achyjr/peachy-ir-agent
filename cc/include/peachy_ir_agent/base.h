#pragma once
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

#define CHECK(cond)               \
  if (!(cond)) {                  \
    LOG(INFO) << "Check Failed."; \
    std::exit(1);                 \
  }

#define ABORT(format, ...)               \
  {                                      \
    char buf[1024];                      \
    sprintf(buf, format, ##__VA_ARGS__); \
    LOG(INFO) << buf;                    \
  }                                      \
  std::exit(1);
