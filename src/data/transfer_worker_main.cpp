// stratakv_transfer_worker (Phase E4, ADR 0002 isolated mode): the
// separate transfer process. Owns the CPU tier BYTES (its own
// allocation under the shared id space), opens the engine's GPU
// buffers via cudaIpcOpenMemHandle, and runs a CudaStreamTransport;
// forwards op completions to the engine over the channel.
//
// Spawned by IpcTransportClient (fd passed by number); never meant to
// be run by hand beyond debugging. Build: STRATAKV_GPU.

#include <cuda_runtime.h>

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "stratakv/common/exceptions.h"
#include "stratakv/common/model_layout.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/data/registry.h"
#include "stratakv/data/tier_table.h"
#include "stratakv/data/transport.h"
#include "stratakv/data/transport_cuda.h"
#include "ipc_protocol.h"

namespace {

using stratakv::common::BufferView;
using stratakv::common::DeviceKind;
using stratakv::common::FullAttnSpec;
using stratakv::common::LayerGroup;
using stratakv::common::LinearStateSpec;
using stratakv::common::MlaSpec;
using stratakv::common::ModelLayout;
using stratakv::common::SlidingWindowSpec;
using stratakv::data::CpuTier;
using stratakv::data::CpuTierConfig;
using stratakv::data::CudaStreamTransport;
using stratakv::data::DeviceMemoryRegistry;
using stratakv::data::GroupDescriptor;
using stratakv::data::GroupLayout;
using stratakv::data::HostBacking;
using stratakv::data::IpcHandle;
using stratakv::data::TierTable;
using stratakv::control::CompletionSink;
using stratakv::control::TransferOp;

using stratakv::data::ipc_protocol::MsgType;
using stratakv::data::ipc_protocol::Received;
using stratakv::data::ipc_protocol::Status;
using stratakv::data::ipc_protocol::Writer;

[[noreturn]] void Die(const std::string& what) {
  std::fprintf(stderr, "transfer_worker: %s\n", what.c_str());
  std::exit(3);
}

/// \brief Serialize a CUDA/IPC failure for the engine (it maps
/// "CUDA IPC unsupported" to a recognizable skip condition).
Status ClassifyError(cudaError_t err) {
  return err == cudaErrorNotSupported ? Status::kIpcUnsupported
                                      : Status::kFailed;
}

void SendStatus(int fd, Status status, const std::string& msg) {
  Writer w;
  w.U64(static_cast<std::uint64_t>(status));
  w.Str(msg);
  try {
    stratakv::data::ipc_protocol::SendMessage(fd, MsgType::kReady, w.bytes);
  } catch (const std::exception&) {
    Die("status send failed (engine gone)");
  }
}

void SendFatal(int fd, const std::string& msg) {
  Writer w;
  w.Str(msg);
  try {
    stratakv::data::ipc_protocol::SendMessage(fd, MsgType::kError, w.bytes);
  } catch (const std::exception&) {
    // Engine gone; nothing more to do.
  }
  std::exit(4);
}

/// \brief Forwards the local transport's completions to the engine.
struct SocketSink final : public CompletionSink {
  int fd;
  std::mutex mu;

  explicit SocketSink(int f) : fd(f) {}

  void OnOpCompleted(std::uint64_t op_id, bool ok) override {
    Writer w;
    w.U32(1);
    w.U64(op_id);
    w.U8(ok ? 1 : 0);
    std::lock_guard<std::mutex> lock(mu);
    try {
      stratakv::data::ipc_protocol::SendMessage(fd, MsgType::kCompletions, w.bytes);
    } catch (const std::exception&) {
      // Engine gone: exit quietly (the engine failed our ops already).
      std::exit(0);
    }
  }
};

struct WorkerState {
  std::unique_ptr<ModelLayout> layout;
  CpuTierConfig tier_config;
  DeviceMemoryRegistry registry;
  TierTable table;
  std::unique_ptr<CpuTier> tier;
  std::unique_ptr<SocketSink> sink;
  std::unique_ptr<CudaStreamTransport> transport;
  std::vector<std::pair<DeviceMemoryRegistry::RegistrationId,
                        std::uint32_t>>
      pending_registrations;

  void BuildTransport() {
    if (transport != nullptr) {
      return;
    }
    tier = std::make_unique<CpuTier>(*layout, tier_config);
    table.AddCpuTier(0, *layout, *tier);  // Its own bytes, same ids.
    transport = std::make_unique<CudaStreamTransport>(std::move(table));
  }
};

void HandleConfigure(WorkerState& state, int fd,
                     const Received& message) {
  try {
    stratakv::data::ipc_protocol::Reader r{message.payload.data(), message.payload.size()};
    if (r.U32() != stratakv::data::ipc_protocol::kVersion) {
      SendStatus(fd, Status::kFailed, "protocol version mismatch");
      return;
    }
    std::vector<LayerGroup> groups;
    const std::uint32_t num_groups = r.U32();
    groups.reserve(num_groups);
    for (std::uint32_t g = 0; g < num_groups; ++g) {
      LayerGroup group;
      group.name = r.Str();
      const std::uint32_t num_layers = r.U32();
      for (std::uint32_t l = 0; l < num_layers; ++l) {
        group.layer_indices.push_back(r.I32());
      }
      group.tokens_per_block = r.I32();
      const std::uint8_t kind = r.U8();
      switch (kind) {
        case 0:
          group.spec = FullAttnSpec{r.I32(), r.I32()};
          break;
        case 1:
          group.spec = MlaSpec{r.I32(), r.I32()};
          break;
        case 2: {
          const int heads = r.I32();
          const int head_size = r.I32();
          const int window = r.I32();
          group.spec = SlidingWindowSpec{heads, head_size, window};
          break;
        }
        default:
          group.spec = LinearStateSpec{r.U64()};
          break;
      }
      groups.push_back(std::move(group));
    }
    state.layout = std::make_unique<ModelLayout>(
        ModelLayout::FromGroups(std::move(groups)));
    const std::uint8_t backing = r.U8();
    state.tier_config.blocks_per_group = r.U32();
    state.tier_config.state_objects_per_group = r.U32();
    state.tier_config.alignment = r.U64();
    switch (backing) {
      case 1: state.tier_config.backing = HostBacking::kPinned; break;
      case 2: state.tier_config.backing = HostBacking::kRegistered; break;
      default: state.tier_config.backing = HostBacking::kPageable; break;
    }
    SendStatus(fd, Status::kOk, "configured");
  } catch (const std::exception& e) {
    SendStatus(fd, Status::kFailed, e.what());
  }
}

void HandleRegisterGroup(WorkerState& state, int fd,
                         const Received& message) {
  if (state.layout == nullptr) {
    SendStatus(fd, Status::kFailed, "registered before configure");
    return;
  }
  try {
    stratakv::data::ipc_protocol::Reader r{message.payload.data(), message.payload.size()};
    GroupDescriptor descriptor;
    descriptor.group_name = r.Str();
    descriptor.group = r.U32();
    descriptor.layout = r.U8() == 1 ? GroupLayout::kBlockwise
                                    : GroupLayout::kLayerwise;
    descriptor.device_id = r.I32();
    const std::uint32_t capacity = static_cast<std::uint32_t>(r.U64());
    const std::uint32_t num_views = r.U32();
    for (std::uint32_t v = 0; v < num_views; ++v) {
      const std::uint64_t size_bytes = r.U64();
      cudaIpcMemHandle_t handle{};
      const std::byte* payload = message.payload.data() + r.at;
      r.Need(sizeof(handle));
      std::memcpy(&handle, payload, sizeof(handle));
      r.at += sizeof(handle);
      void* dev = nullptr;
      const cudaError_t err = cudaIpcOpenMemHandle(
          &dev, handle, cudaIpcMemLazyEnablePeerAccess);
      if (err != cudaSuccess) {
        SendStatus(fd, ClassifyError(err),
                   std::string("CUDA IPC unsupported on this platform "
                               "or open failed: ") +
                       cudaGetErrorString(err));
        return;
      }
      BufferView view;
      view.base = dev;
      view.kind = DeviceKind::kGpu;
      view.device_id = descriptor.device_id;
      view.size_bytes = size_bytes;
      descriptor.per_layer.push_back(view);
    }
    const auto registration =
        state.registry.RegisterGroup(std::move(descriptor));
    // The TierTable snapshot travels with the transport; buffer groups
    // and build it at the first Enqueue (all groups registered by
    // then — Enqueue only happens after RegisterGpuGroup returns).
    state.pending_registrations.push_back({registration, capacity});
    SendStatus(fd, Status::kOk, "registered");
  } catch (const std::exception& e) {
    SendStatus(fd, Status::kFailed, e.what());
  }
}

void HandleEnqueue(WorkerState& state, int fd, const Received& message) {
  if (state.layout == nullptr || state.pending_registrations.empty()) {
    SendFatal(fd, "enqueue before configure/registration");
  }
  try {
    if (state.transport == nullptr) {
      for (const auto& [registration, capacity] :
           state.pending_registrations) {
        state.table.AddGpuTier(1, state.registry, registration,
                               *state.layout, capacity);
      }
      state.BuildTransport();
    }
    stratakv::data::ipc_protocol::Reader r{message.payload.data(), message.payload.size()};
    const std::uint32_t num_ops = r.U32();
    std::vector<TransferOp> ops;
    ops.reserve(num_ops);
    for (std::uint32_t i = 0; i < num_ops; ++i) {
      TransferOp op;
      op.op_id = r.U64();
      op.kind = static_cast<TransferOp::Kind>(r.U8());
      op.src_tier = r.U32();
      op.dst_tier = r.U32();
      op.src_group = r.U32();
      op.dst_group = r.U32();
      const std::uint32_t n = r.U32();
      op.src_blocks.resize(n);
      for (std::uint32_t k = 0; k < n; ++k) {
        op.src_blocks[k] = r.U32();
      }
      op.dst_blocks.resize(n);
      for (std::uint32_t k = 0; k < n; ++k) {
        op.dst_blocks[k] = r.U32();
      }
      op.src_object = r.U32();
      op.dst_object = r.U32();
      ops.push_back(std::move(op));
    }
    state.transport->Enqueue(std::move(ops), *state.sink);
  } catch (const stratakv::LogicError& e) {
    // Batch validation failure: the engine must hear about every op.
    SendFatal(fd, std::string("enqueue rejected: ") + e.what());
  } catch (const std::exception& e) {
    SendFatal(fd, e.what());
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3 || std::strcmp(argv[1], "--fd") != 0) {
    std::fprintf(stderr, "usage: %s --fd <socket-fd>\n", argv[0]);
    return 2;
  }
  const int fd = std::atoi(argv[2]);

  WorkerState state;
  state.sink = std::make_unique<SocketSink>(fd);

  while (true) {
    Received message;
    try {
      message = stratakv::data::ipc_protocol::ReceiveMessage(fd, -1);
    } catch (const std::exception&) {
      return 0;  // Engine closed the channel: clean exit.
    }
    switch (message.type) {
      case MsgType::kConfigure:
        HandleConfigure(state, fd, message);
        break;
      case MsgType::kRegisterGroup:
        HandleRegisterGroup(state, fd, message);
        break;
      case MsgType::kEnqueueBatch:
        HandleEnqueue(state, fd, message);
        break;
      case MsgType::kShutdown: {
        if (state.transport != nullptr) {
          stratakv::data::ipc_protocol::Reader r{message.payload.data(),
                                 message.payload.size()};
          const std::uint64_t grace = r.U64();
          state.transport->DrainPending(
              std::chrono::milliseconds(grace));
        }
        return 0;
      }
      default:
        SendFatal(fd, "unexpected message type");
        return 4;
    }
  }
}
