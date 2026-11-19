// Wire protocol between the engine process's IpcTransportClient and
// the stratakv_transfer_worker (Phase E4, ADR 0002 isolated mode).
// Internal header shared by src/data/ipc_transport.cpp and
// src/data/transfer_worker_main.cpp — never installed.
//
// Framing: every message = [u32 payload_len][u32 MsgType][payload].
// All integers little-endian (both sides are x86_64/Linux today; the
// reader validates bounds and throws on malformed input — the worker
// answers kError and keeps running rather than crashing).

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace stratakv::data::ipc_protocol {

constexpr std::uint32_t kVersion = 1;

enum class MsgType : std::uint32_t {
  kConfigure = 1,    ///< Engine -> worker: layout + CPU tier config.
  kRegisterGroup = 2,  ///< Engine -> worker: one GPU group's IPC handles.
  kReady = 3,        ///< Worker -> engine: ack [u64 status][u32 len][msg].
  kEnqueueBatch = 4,   ///< Engine -> worker: ops to execute.
  kCompletions = 5,    ///< Worker -> engine: [u32 n]{u64 op_id, u8 ok}.
  kShutdown = 6,     ///< Engine -> worker: [u64 grace_ms].
  kError = 7,        ///< Worker -> engine: fatal, [u32 len][msg].
};

/// \brief kReady status codes (worker-side outcomes).
enum class Status : std::uint64_t {
  kOk = 0,
  kIpcUnsupported = 1,  ///< cudaIpcOpenMemHandle refused (e.g. WSL2).
  kFailed = 2,
};

/// \brief Bounded little-endian writer.
struct Writer {
  std::vector<std::byte> bytes;

  void U8(std::uint8_t v) {
    const std::byte* raw = reinterpret_cast<const std::byte*>(&v);
    bytes.insert(bytes.end(), raw, raw + 1);
  }
  void U32(std::uint32_t v) {
    const std::byte* raw = reinterpret_cast<const std::byte*>(&v);
    bytes.insert(bytes.end(), raw, raw + 4);
  }
  void U64(std::uint64_t v) {
    const std::byte* raw = reinterpret_cast<const std::byte*>(&v);
    bytes.insert(bytes.end(), raw, raw + 8);
  }
  void I32(std::int32_t v) { U32(static_cast<std::uint32_t>(v)); }
  void Str(const std::string& s) {
    U32(static_cast<std::uint32_t>(s.size()));
    const std::byte* raw = reinterpret_cast<const std::byte*>(s.data());
    bytes.insert(bytes.end(), raw, raw + s.size());
  }
  void Blob(const void* data, std::size_t n) {
    const std::byte* raw = static_cast<const std::byte*>(data);
    bytes.insert(bytes.end(), raw, raw + n);
  }
};

/// \brief Bounds-checked little-endian reader.
struct Reader {
  const std::byte* data;
  std::size_t size;
  std::size_t at = 0;

  void Need(std::size_t n) const {
    if (at + n > size) {
      throw std::runtime_error("ipc protocol: truncated message");
    }
  }
  std::uint8_t U8() {
    Need(1);
    return static_cast<std::uint8_t>(data[at++]);
  }
  std::uint32_t U32() {
    Need(4);
    std::uint32_t v;
    std::memcpy(&v, data + at, 4);
    at += 4;
    return v;
  }
  std::uint64_t U64() {
    Need(8);
    std::uint64_t v;
    std::memcpy(&v, data + at, 8);
    at += 8;
    return v;
  }
  std::int32_t I32() {
    return static_cast<std::int32_t>(U32());
  }
  std::string Str() {
    const std::uint32_t n = U32();
    Need(n);
    std::string out(reinterpret_cast<const char*>(data + at), n);
    at += n;
    return out;
  }
};

/// \brief Framed send/recv over a stream socket fd (blocking, with
/// poll-based deadline for reads). Throws std::runtime_error on I/O
/// failure or timeout — callers translate to their failure policy.
void SendMessage(int fd, MsgType type, const std::vector<std::byte>& payload);

struct Received {
  MsgType type;
  std::vector<std::byte> payload;
};

/// \brief Blocking read of one framed message; deadline_ms < 0 waits
/// forever.
Received ReceiveMessage(int fd, long deadline_ms);

}  // namespace stratakv::data::ipc_protocol
