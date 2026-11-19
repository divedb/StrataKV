// IpcTransportClient (Phase E4, ADR 0002 isolated mode): engine-side
// transport. Spawn/handshake in the constructor (synchronous), a
// receiver thread for completions, inline failure of in-flight ops
// when the worker dies (fault tolerance — the engine must survive).

#include "stratakv/data/ipc_transport.h"

#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "ipc_protocol.h"
#include "stratakv/data/transport_cuda.h"

namespace stratakv::data {

namespace {

using ipc_protocol::MsgType;
using ipc_protocol::Received;
using ipc_protocol::Status;
using ipc_protocol::Writer;

constexpr char kUnsupportedPrefix[] = "CUDA IPC unsupported on this platform";

std::string ErrnoMessage(const char* what) {
  return std::string(what) + ": " + std::strerror(errno);
}

/// \brief Copy of the model layout on the wire (the worker rebuilds
/// ModelLayout from it; keeps the protocol explicit over sending C++
/// types).
void WriteLayout(Writer& w, const common::ModelLayout& layout) {
  w.U32(ipc_protocol::kVersion);
  w.U32(static_cast<std::uint32_t>(layout.num_groups()));
  for (std::size_t g = 0; g < layout.num_groups(); ++g) {
    const common::LayerGroup& group = layout.Group(static_cast<int>(g));
    w.Str(group.name);
    w.U32(static_cast<std::uint32_t>(group.layer_indices.size()));
    for (int index : group.layer_indices) {
      w.I32(index);
    }
    w.I32(group.tokens_per_block);
    if (const auto* full = std::get_if<common::FullAttnSpec>(&group.spec)) {
      w.U8(0);
      w.I32(full->num_kv_heads);
      w.I32(full->head_size);
    } else if (const auto* mla = std::get_if<common::MlaSpec>(&group.spec)) {
      w.U8(1);
      w.I32(mla->kv_lora_rank);
      w.I32(mla->rope_dim);
    } else if (const auto* swa =
                   std::get_if<common::SlidingWindowSpec>(&group.spec)) {
      w.U8(2);
      w.I32(swa->num_kv_heads);
      w.I32(swa->head_size);
      w.I32(swa->window_blocks);
    } else {
      const auto& linear = std::get<common::LinearStateSpec>(group.spec);
      w.U8(3);
      w.U64(linear.state_bytes);
    }
  }
}

}  // namespace

struct IpcTransportClient::Impl {
  Config config;
  int fd = -1;
  pid_t worker = -1;

  mutable std::mutex mu;  // Guards socket writes + pending map + flags.
  std::map<std::uint64_t, control::CompletionSink*> pending;
  bool dead = false;
  std::string dead_reason;

  std::thread receiver;
  bool receiver_started = false;

  /// \brief Receiver starts at the FIRST Enqueue, after all
  /// registrations: setup acks are read synchronously on the caller's
  /// thread, so a running receiver could never race them away.
  void EnsureReceiver() {
    if (receiver_started) {
      return;
    }
    receiver_started = true;
    receiver = std::thread([this] { ReceiverLoop(); });
  }

  /// \brief Completion pump: socket -> CompletionSink callbacks.
  void ReceiverLoop();

  ~Impl() {
    if (receiver.joinable()) {
      receiver.join();
    }
    if (fd >= 0) {
      (void)close(fd);
    }
  }

  void MarkDead(const std::string& reason) {
    std::vector<std::pair<std::uint64_t, control::CompletionSink*>> fail;
    {
      std::lock_guard<std::mutex> lock(mu);
      if (dead) {
        return;
      }
      dead = true;
      dead_reason = reason;
      for (const auto& [op_id, sink] : pending) {
        fail.emplace_back(op_id, sink);
      }
      pending.clear();
    }
    for (const auto& [op_id, sink] : fail) {
      sink->OnOpCompleted(op_id, /*ok=*/false);
    }
  }
};

IpcTransportClient::IpcTransportClient(const common::ModelLayout& layout,
                                       CpuTierConfig tier_config)
    : IpcTransportClient(layout, std::move(tier_config), Config()) {}

IpcTransportClient::IpcTransportClient(const common::ModelLayout& layout,
                                       CpuTierConfig tier_config,
                                       Config config)
    : impl_(std::make_unique<Impl>()) {
  impl_->config = config;

  // Resolve the worker binary: explicit path, env, exe-relative.
  std::string worker_path = config.worker_path;
  if (worker_path.empty()) {
    const char* env = std::getenv("STRATAKV_TRANSFER_WORKER");
    if (env != nullptr) {
      worker_path = env;
    }
  }
  if (worker_path.empty()) {
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) {
      throw LogicError(ErrnoMessage("IpcTransportClient: /proc/self/exe"));
    }
    self[n] = '\0';
    std::string dir(self);
    const std::size_t slash = dir.rfind('/');
    dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    worker_path = dir + "/stratakv_transfer_worker";
  }

  int sv[2];
  if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
    throw LogicError(ErrnoMessage("IpcTransportClient: socketpair"));
  }
  const pid_t pid = fork();
  if (pid < 0) {
    (void)close(sv[0]);
    (void)close(sv[1]);
    throw LogicError(ErrnoMessage("IpcTransportClient: fork"));
  }
  if (pid == 0) {
    // Child: worker keeps sv[1] as fd number (no CLOEXEC set).
    (void)close(sv[0]);
    char fd_arg[16];
    std::snprintf(fd_arg, sizeof(fd_arg), "%d", sv[1]);
    char* argv[] = {const_cast<char*>(worker_path.c_str()),
                    const_cast<char*>("--fd"), fd_arg, nullptr};
    execv(worker_path.c_str(), argv);
    // exec failed: the parent's ack read fails loudly.
    _exit(127);
  }
  (void)close(sv[1]);
  impl_->fd = sv[0];
  impl_->worker = pid;

  const long deadline_ms =
      static_cast<long>(impl_->config.ack_timeout.count());

  auto ack = [&](const char* stage) {
    Received ready;
    try {
      ready = ipc_protocol::ReceiveMessage(impl_->fd, deadline_ms);
    } catch (const std::exception& e) {
      int status = 0;
      (void)waitpid(pid, &status, 0);
      throw LogicError(std::string("IpcTransportClient: ") + stage +
                       " failed: " + e.what());
    }
    if (ready.type != MsgType::kReady) {
      throw LogicError("IpcTransportClient: unexpected message during " +
                       std::string(stage));
    }
    ipc_protocol::Reader r{ready.payload.data(), ready.payload.size()};
    const std::uint64_t status = r.U64();
    const std::string msg = r.Str();
    if (status != static_cast<std::uint64_t>(Status::kOk)) {
      throw LogicError("IpcTransportClient: worker rejected " +
                       std::string(stage) + ": " + msg);
    }
  };

  // Configure: layout + CPU tier config (the worker's own allocation).
  {
    Writer w;
    WriteLayout(w, layout);
    w.U8(static_cast<std::uint8_t>(tier_config.backing));
    w.U32(tier_config.blocks_per_group);
    w.U32(tier_config.state_objects_per_group);
    w.U64(tier_config.alignment);
    try {
      ipc_protocol::SendMessage(impl_->fd, MsgType::kConfigure, w.bytes);
    } catch (const std::exception& e) {
      throw LogicError(std::string("IpcTransportClient: configure: ") +
                       e.what());
    }
    ack("configure");
  }
}

IpcTransportClient::~IpcTransportClient() {
  DrainPending(std::chrono::milliseconds(0));
}

void IpcTransportClient::Impl::ReceiverLoop() {
  while (true) {
    Received message;
    try {
      message = ipc_protocol::ReceiveMessage(fd, -1);
    } catch (const std::exception& e) {
      MarkDead(e.what());
      return;
    }
    switch (message.type) {
      case MsgType::kCompletions: {
        ipc_protocol::Reader r{message.payload.data(), message.payload.size()};
        const std::uint32_t n = r.U32();
        for (std::uint32_t i = 0; i < n; ++i) {
          const std::uint64_t op_id = r.U64();
          const bool ok = r.U8() != 0;
          control::CompletionSink* sink = nullptr;
          {
            std::lock_guard<std::mutex> lock(mu);
            const auto it = pending.find(op_id);
            if (it != pending.end()) {
              sink = it->second;
              pending.erase(it);
            }
          }
          if (sink != nullptr) {
            sink->OnOpCompleted(op_id, ok);
          }
        }
        break;
      }
      case MsgType::kError: {
        ipc_protocol::Reader r{message.payload.data(), message.payload.size()};
        MarkDead("worker error: " + r.Str());
        return;
      }
      default:
        MarkDead("worker sent an unexpected message");
        return;
    }
  }
}

void IpcTransportClient::RegisterGpuGroup(const GroupDescriptor& descriptor,
                                          std::uint32_t capacity_blocks) {
  CudaIpcHandleSource exporter;
  Writer w;
  w.Str(descriptor.group_name);
  w.U32(descriptor.group);
  w.U8(descriptor.layout == GroupLayout::kBlockwise ? 1 : 0);
  w.I32(descriptor.device_id);
  w.U64(capacity_blocks);
  w.U32(static_cast<std::uint32_t>(descriptor.per_layer.size()));
  for (const common::BufferView& view : descriptor.per_layer) {
    const IpcHandle handle = exporter.Export(view);  // Throws loudly.
    w.U64(view.size_bytes);
    w.Blob(handle.opaque.data(), handle.opaque.size());
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->dead) {
      throw LogicError("IpcTransportClient: worker is down: " +
                       impl_->dead_reason);
    }
    try {
      ipc_protocol::SendMessage(impl_->fd, MsgType::kRegisterGroup,
                                w.bytes);
    } catch (const std::exception& e) {
      impl_->MarkDead(e.what());
      throw LogicError(std::string("IpcTransportClient: register: ") +
                       e.what());
    }
  }
  // Synchronous ack read (safe: the receiver thread is not running
  // until the first Enqueue).
  Received ready;
  try {
    ready = ipc_protocol::ReceiveMessage(
        impl_->fd, static_cast<long>(impl_->config.ack_timeout.count()));
  } catch (const std::exception& e) {
    impl_->MarkDead(e.what());
    throw LogicError(std::string("IpcTransportClient: register ack: ") +
                     e.what());
  }
  if (ready.type != MsgType::kReady) {
    impl_->MarkDead("unexpected message during register");
    throw LogicError("IpcTransportClient: unexpected register ack");
  }
  ipc_protocol::Reader r{ready.payload.data(), ready.payload.size()};
  const std::uint64_t status = r.U64();
  const std::string msg = r.Str();
  if (status != static_cast<std::uint64_t>(Status::kOk)) {
    impl_->MarkDead("register rejected: " + msg);
    throw LogicError("IpcTransportClient: worker rejected register: " + msg);
  }
}

void IpcTransportClient::Enqueue(std::vector<control::TransferOp> batch,
                                 control::CompletionSink& sink) {
  if (batch.empty()) {
    return;
  }
  Writer w;
  w.U32(static_cast<std::uint32_t>(batch.size()));
  for (const control::TransferOp& op : batch) {
    w.U64(op.op_id);
    w.U8(static_cast<std::uint8_t>(op.kind));
    w.U32(op.src_tier);
    w.U32(op.dst_tier);
    w.U32(op.src_group);
    w.U32(op.dst_group);
    w.U32(static_cast<std::uint32_t>(op.src_blocks.size()));
    for (common::BlockId id : op.src_blocks) {
      w.U32(id);
    }
    for (common::BlockId id : op.dst_blocks) {
      w.U32(id);
    }
    w.U32(op.src_object);
    w.U32(op.dst_object);
  }
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->dead) {
      // Fail inline: exactly once per op, per contract.
      for (const control::TransferOp& op : batch) {
        sink.OnOpCompleted(op.op_id, /*ok=*/false);
      }
      return;
    }
    for (const control::TransferOp& op : batch) {
      impl_->pending[op.op_id] = &sink;
    }
    impl_->EnsureReceiver();  // First Enqueue starts completion pickup.
    try {
      ipc_protocol::SendMessage(impl_->fd, MsgType::kEnqueueBatch, w.bytes);
    } catch (const std::exception& e) {
      for (const control::TransferOp& op : batch) {
        impl_->pending.erase(op.op_id);
        sink.OnOpCompleted(op.op_id, /*ok=*/false);
      }
      impl_->dead = true;
      impl_->dead_reason = e.what();
    }
  }
}

void IpcTransportClient::DrainPending(std::chrono::milliseconds grace) {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    if (impl_->fd >= 0 && !impl_->dead) {
      Writer w;
      w.U64(static_cast<std::uint64_t>(grace.count()));
      try {
        ipc_protocol::SendMessage(impl_->fd, MsgType::kShutdown, w.bytes);
      } catch (const std::exception&) {
        // Worker already gone; fall through to reap.
      }
      (void)shutdown(impl_->fd, SHUT_WR);
    }
  }
  const auto deadline =
      std::chrono::steady_clock::now() + grace;
  if (impl_->worker > 0) {
    int status = 0;
    while (waitpid(impl_->worker, &status, WNOHANG) == 0) {
      if (std::chrono::steady_clock::now() >= deadline) {
        (void)kill(impl_->worker, SIGKILL);
        (void)waitpid(impl_->worker, &status, 0);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    impl_->worker = -1;
  }
  impl_->MarkDead("drained");
  std::unique_lock<std::mutex> lock(impl_->mu);
  if (impl_->receiver.joinable()) {
    // The receiver exits once the socket closes (worker exit); do not
    // hold mu while joining (ReceiverLoop takes it via MarkDead).
    lock.unlock();
    impl_->receiver.join();
  }
}

std::size_t IpcTransportClient::num_in_flight() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  return impl_->pending.size();
}

pid_t IpcTransportClient::worker_pid() const { return impl_->worker; }

bool IpcTransportClient::IsIpcUnsupported(const std::string& message) {
  return message.find("CUDA IPC unsupported") != std::string::npos ||
         message.find("cudaErrorNotSupported") != std::string::npos;
}

}  // namespace stratakv::data
