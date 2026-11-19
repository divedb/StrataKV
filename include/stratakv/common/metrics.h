#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>

namespace stratakv::common {

/// \brief Bounded, thread-safe host wall-clock latency histogram.
class LatencyHistogram {
 public:
  struct Snapshot {
    static constexpr std::array<std::uint64_t, 7> bounds_us = {
        1, 10, 100, 1000, 10000, 100000, 1000000};
    std::array<std::uint64_t, 8> buckets{};
    std::uint64_t count = 0;
    double sum_seconds = 0;
  };

  /// \brief Record one observation; buckets are disjoint, last is infinity.
  void Observe(std::chrono::steady_clock::duration elapsed) {
    const double us = std::chrono::duration<double, std::micro>(elapsed).count();
    const auto bucket = std::lower_bound(Snapshot::bounds_us.begin(),
                                        Snapshot::bounds_us.end(), us);
    std::lock_guard<std::mutex> lock(mu_);
    ++snapshot_.buckets[static_cast<std::size_t>(bucket - Snapshot::bounds_us.begin())];
    ++snapshot_.count;
    snapshot_.sum_seconds += us / 1000000.0;
  }

  /// \brief Copy a coherent histogram without resetting it.
  Snapshot Read() const {
    std::lock_guard<std::mutex> lock(mu_);
    return snapshot_;
  }

 private:
  mutable std::mutex mu_;
  Snapshot snapshot_;
};

/// \brief Observe a scope on every exit, including exceptions.
class ScopedLatency {
 public:
  explicit ScopedLatency(LatencyHistogram& histogram)
      : histogram_(histogram), start_(std::chrono::steady_clock::now()) {}
  ~ScopedLatency() { histogram_.Observe(std::chrono::steady_clock::now() - start_); }
  ScopedLatency(const ScopedLatency&) = delete;
  ScopedLatency& operator=(const ScopedLatency&) = delete;
 private:
  LatencyHistogram& histogram_;
  std::chrono::steady_clock::time_point start_;
};
}  // namespace stratakv::common
