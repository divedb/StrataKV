// shim_main.cpp — test runner for the offline bootstrap shim. Compiled
// only with -DSTRATAKV_GTEST_SHIM=ON; the real build links gtest_main.
#include "gtest_shim.h"

int main() {
  int failed = 0;
  int total = 0;
  for (const auto& test : ::stratakv_gtest_shim::Registry()) {
    ++total;
    const int before = ::stratakv_gtest_shim::FailureCount();
    try {
      test.body();
    } catch (const std::exception& error) {
      ++::stratakv_gtest_shim::FailureCount();
      std::fprintf(stderr, "UNCAUGHT EXCEPTION in %s: %s\n", test.name.c_str(),
                   error.what());
    } catch (...) {
      ++::stratakv_gtest_shim::FailureCount();
      std::fprintf(stderr, "UNCAUGHT EXCEPTION in %s\n", test.name.c_str());
    }
    if (::stratakv_gtest_shim::FailureCount() > before) {
      ++failed;
      std::printf("[  FAILED  ] %s\n", test.name.c_str());
    } else {
      std::printf("[       OK ] %s\n", test.name.c_str());
    }
  }
  std::printf("[==========] %d tests, %d failures\n", total, failed);
  return failed == 0 ? 0 : 1;
}
