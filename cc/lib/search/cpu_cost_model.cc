#include "peachy_ir_agent/search/cpu_cost_model.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

#include "peachy_ir_agent/codegen/codegen.h"
#include "peachy_ir_agent/codegen/cpp_gen.h"
#include "peachy_ir_agent/config.h"

namespace peachyir {
namespace {
static constexpr char kResFilename[] = "/tmp/peachyir.prof";
namespace fs = std::filesystem;
}  // namespace

static constexpr char kBenchmarkTemplate[] = R"(
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>

static constexpr size_t kNumRuns = 25;
static constexpr char kResFilename[] = "%RES_FILENAME%";

%KERNEL_DEF%

int main(int argc, char** argv) {
  %ALLOCATE_BUFFERS%
  %INIT_BUFFERS%
  double avg_runtime_ns = 0.0;
  for (int run = 1; run <= kNumRuns; ++run) {
    auto start = std::chrono::high_resolution_clock::now();
    %KERNEL_INVOKE%
    auto end = std::chrono::high_resolution_clock::now();
    auto runtime_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();
    avg_runtime_ns += (runtime_ns - avg_runtime_ns) / run;

    %RESET_BUFFERS%
  }

  std::ofstream file(kResFilename);
  file << avg_runtime_ns;
  file.close();
  return 0;
}
)";

float CpuCostModel::cost(FunctionNodePtr ir) {
  std::string cpp_kernel = codegen(ir);
  std::string cpp_alloc = CppAllocGen::apply(ir);
  std::string cpp_invoke = CppInvokeGen::apply(ir);
  std::string cpp_init = CppInitGen::apply(ir);
  std::string cpp_reset = CppResetGen::apply(ir);

  // Fill in code template.
  std::string bm_code =
      std::regex_replace(std::string(kBenchmarkTemplate),
                         std::regex("%ALLOCATE_BUFFERS%"), cpp_alloc);
  bm_code = std::regex_replace(bm_code, std::regex("%INIT_BUFFERS%"), cpp_init);
  bm_code =
      std::regex_replace(bm_code, std::regex("%RESET_BUFFERS%"), cpp_reset);
  bm_code =
      std::regex_replace(bm_code, std::regex("%KERNEL_INVOKE%"), cpp_invoke);
  bm_code = std::regex_replace(bm_code, std::regex("%KERNEL_DEF%"), cpp_kernel);
  bm_code =
      std::regex_replace(bm_code, std::regex("%RES_FILENAME%"), kResFilename);

  std::string bm_path = fs::path(PROJECT_ROOT_DIR) / "__autogen__" / "main.cc";
  std::ofstream file(bm_path);
  file << bm_code;
  file.close();

  std::string bin_path = fs::path(PROJECT_ROOT_DIR) / "__autogen__" / "main";

  // Compile benchmark.
  std::string compile_cmd =
      "g++ -O3 -fopenmp -funroll-loops -std=c++17 -march=native " + bm_path +
      " -o " + bin_path;
  std::system(compile_cmd.c_str());

  // Run benchmark.
  std::system(bin_path.c_str());

  std::ifstream res_file(kResFilename);
  std::string time;
  std::getline(res_file, time);

  return std::atof(time.c_str());
}

}  // namespace peachyir
