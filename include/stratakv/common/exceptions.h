#pragma once

#include <stdexcept>
#include <string>

namespace stratakv {

/// \brief Storage exhaustion (FlexKV `NotEnoughSpaceError` parity).
///
/// Thrown by allocators when a request cannot be satisfied from the
/// remaining free capacity.
class NotEnoughSpaceError : public std::runtime_error {
 public:
  explicit NotEnoughSpaceError(const std::string& what)
      : std::runtime_error(what) {}
};

/// \brief Contract violation detectable at the call site (FlexKV
/// `LogicError` parity).
///
/// Examples: double-release of a pooled id, unlocking an unpinned match,
/// committing an unknown cache entry.
class LogicError : public std::logic_error {
 public:
  explicit LogicError(const std::string& what) : std::logic_error(what) {}
};

}  // namespace stratakv
