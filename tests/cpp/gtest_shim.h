// gtest_shim.h — minimal offline bootstrap subset of the GoogleTest API
// used by StrataKV unit tests.
//
// This shim exists ONLY because the development environment currently has
// no network access to fetch the pinned googletest submodule. It is not a
// GoogleTest replacement: CI and any networked environment must use the
// real pinned submodule (scripts/fetch_third_party.sh). Enable with
// -DSTRATAKV_GTEST_SHIM=ON.
#pragma once

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stratakv_gtest_shim {

struct TestCase {
  std::string name;
  void (*body)();
};

inline std::vector<TestCase>& Registry() {
  static std::vector<TestCase> registry;
  return registry;
}

inline int& FailureCount() {
  static int count = 0;
  return count;
}

template <typename T>
std::string Print(const T& value) {
  std::ostringstream os;
  os << value;
  return os.str();
}
inline std::string Print(bool value) { return value ? "true" : "false"; }
inline std::string Print(const char* value) {
  return value != nullptr ? value : "(null)";
}
inline std::string Print(std::nullptr_t) { return "nullptr"; }

}  // namespace stratakv_gtest_shim

#define STRATAKV_TEST_REPORT(file, line, expr, lhs, rhs, fatal)            \
  do {                                                                     \
    ++::stratakv_gtest_shim::FailureCount();                               \
    std::fprintf(stderr, "%s:%d: FAILURE: %s (%s vs %s)\n", file, line,    \
                 expr, ::stratakv_gtest_shim::Print(lhs).c_str(),          \
                 ::stratakv_gtest_shim::Print(rhs).c_str());               \
    if (fatal) return;                                                     \
  } while (false)

#define STRATAKV_TEST_BINARY(lhs, op, rhs, fatal)                          \
  do {                                                                     \
    if (!((lhs)op(rhs))) {                                                 \
      STRATAKV_TEST_REPORT(__FILE__, __LINE__, #lhs " " #op " " #rhs, lhs, \
                           rhs, fatal);                                    \
    }                                                                      \
  } while (false)

#define TEST(suite, name)                                       \
  static void suite##_##name##_body();                          \
  [[maybe_unused]] static const bool suite##_##name##_registered = [] {      \
    ::stratakv_gtest_shim::Registry().push_back(                \
        {#suite "." #name, &suite##_##name##_body});            \
    return true;                                                \
  }();                                                          \
  static void suite##_##name##_body()

#define EXPECT_TRUE(x) STRATAKV_TEST_BINARY(!!(x), ==, true, false)
#define EXPECT_FALSE(x) STRATAKV_TEST_BINARY(!!(x), ==, false, false)
#define EXPECT_EQ(a, b) STRATAKV_TEST_BINARY(a, ==, b, false)
#define EXPECT_NE(a, b) STRATAKV_TEST_BINARY(a, !=, b, false)
#define EXPECT_LT(a, b) STRATAKV_TEST_BINARY(a, <, b, false)
#define EXPECT_LE(a, b) STRATAKV_TEST_BINARY(a, <=, b, false)
#define EXPECT_GT(a, b) STRATAKV_TEST_BINARY(a, >, b, false)
#define EXPECT_GE(a, b) STRATAKV_TEST_BINARY(a, >=, b, false)
#define ASSERT_TRUE(x) STRATAKV_TEST_BINARY(!!(x), ==, true, true)
#define ASSERT_EQ(a, b) STRATAKV_TEST_BINARY(a, ==, b, true)

#define EXPECT_THROW(statement, exception_type)                             \
  do {                                                                      \
    bool stratakv_threw_ = false;                                           \
    try {                                                                   \
      statement;                                                            \
    } catch (const exception_type&) {                                       \
      stratakv_threw_ = true;                                               \
    }                                                                       \
    if (!stratakv_threw_) {                                                 \
      ++::stratakv_gtest_shim::FailureCount();                              \
      std::fprintf(stderr,                                                  \
                   "%s:%d: FAILURE: expected exception %s was not thrown\n",\
                   __FILE__, __LINE__, #exception_type);                    \
    }                                                                       \
  } while (false)
