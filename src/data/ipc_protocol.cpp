// Framed socket I/O for the E4 engine<->worker protocol.

#include "ipc_protocol.h"

#include <arpa/inet.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <stdexcept>

namespace stratakv::data::ipc_protocol {

namespace {

void WriteAll(int fd, const void* data, std::size_t n) {
  const char* at = static_cast<const char*>(data);
  while (n > 0) {
    const ssize_t wrote = ::send(fd, at, n, MSG_NOSIGNAL);
    if (wrote < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("ipc send: ") +
                               std::strerror(errno));
    }
    at += wrote;
    n -= static_cast<std::size_t>(wrote);
  }
}

bool ReadSome(int fd, void* data, std::size_t n, long deadline_ms) {
  // Blocking read of exactly n bytes with a poll deadline; false on
  // clean EOF at a message boundary (n == 0 handled by callers).
  char* at = static_cast<char*>(data);
  while (n > 0) {
    if (deadline_ms >= 0) {
      struct pollfd pfd{};
      pfd.fd = fd;
      pfd.events = POLLIN;
      const int ready = ::poll(&pfd, 1, static_cast<int>(deadline_ms));
      if (ready == 0) {
        throw std::runtime_error("ipc recv: timeout");
      }
      if (ready < 0 && errno != EINTR) {
        throw std::runtime_error(std::string("ipc poll: ") +
                                 std::strerror(errno));
      }
      if (ready == 0 || (ready < 0 && errno == EINTR)) {
        continue;
      }
    }
    const ssize_t got = ::recv(fd, at, n, 0);
    if (got == 0) {
      throw std::runtime_error("ipc recv: connection closed");
    }
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(std::string("ipc recv: ") +
                               std::strerror(errno));
    }
    at += got;
    n -= static_cast<std::size_t>(got);
  }
  return true;
}

}  // namespace

void SendMessage(int fd, MsgType type,
                 const std::vector<std::byte>& payload) {
  const std::uint32_t len =
      static_cast<std::uint32_t>(payload.size() + sizeof(std::uint32_t));
  const std::uint32_t wire_type = static_cast<std::uint32_t>(type);
  WriteAll(fd, &len, sizeof(len));
  WriteAll(fd, &wire_type, sizeof(wire_type));
  if (!payload.empty()) {
    WriteAll(fd, payload.data(), payload.size());
  }
}

Received ReceiveMessage(int fd, long deadline_ms) {
  // Recompute the absolute deadline across both header+payload reads
  // when one is given.
  const auto started = std::chrono::steady_clock::now();
  auto remaining = [started, deadline_ms]() -> long {
    if (deadline_ms < 0) {
      return -1;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    return deadline_ms - static_cast<long>(elapsed.count());
  };
  std::uint32_t len = 0;
  ReadSome(fd, &len, sizeof(len), remaining());
  std::uint32_t wire_type = 0;
  ReadSome(fd, &wire_type, sizeof(wire_type), remaining());
  if (len < sizeof(std::uint32_t)) {
    throw std::runtime_error("ipc recv: malformed length");
  }
  const std::size_t payload_size = len - sizeof(std::uint32_t);
  Received out;
  out.type = static_cast<MsgType>(wire_type);
  out.payload.resize(payload_size);
  if (payload_size > 0) {
    ReadSome(fd, out.payload.data(), payload_size, remaining());
  }
  return out;
}

}  // namespace stratakv::data::ipc_protocol
