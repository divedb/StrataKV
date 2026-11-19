#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <sys/types.h>  // pid_t

#include "stratakv/common/model_layout.h"
#include "stratakv/data/cpu_tier.h"
#include "stratakv/data/registry.h"
#include "stratakv/data/transport.h"

namespace stratakv::data {

/// \brief Engine-side transport for ADR 0002's isolated mode: spawns
/// the stratakv_transfer_worker process and forwards ops to it over a
/// unix socket (Phase E4).
///
/// Division of state (the ADR 0002 experiment's substrate):
/// - Engine process: MetadataPlane (ids), TaskEngine, this client, and
///   the ENGINE-OWNED GPU memory (registered here as usual; handles are
///   exported via cudaIpcGetMemHandle and sent to the worker).
/// - Worker process: the CPU tier BYTES (its own allocation under the
///   shared id space — ids address slots, not pointers, so both sides
///   stay in sync), the IPC-opened GPU mappings, and a full
///   CudaStreamTransport.
///
/// Completion flow: worker's transport events -> kCompletions messages
/// -> this client's receiver thread -> CompletionSink::OnOpCompleted
/// (any executor thread, per the ITransport contract). Requires
/// cross-process CUDA IPC (native Linux; WSL2 refuses the open — the
/// worker reports that and construction fails loudly with a
/// recognizable message so callers can skip on such platforms).
///
/// Worker binary resolution: $STRATAKV_TRANSFER_WORKER, else a binary
/// named stratakv_transfer_worker next to the running executable.
///
/// Thread safety: Enqueue safe from any engine thread; registrations
/// and Shutdown are the owner's to serialize (constructor +
/// DrainPending).
class IpcTransportClient final : public ITransport {
 public:
  struct Config {
    /// \brief Per-message ack timeout for the setup handshake.
    std::chrono::milliseconds ack_timeout{std::chrono::milliseconds(5000)};
    /// \brief Explicit worker binary path (else env/exe-dir lookup).
    std::string worker_path;
  };

  /// \param layout Model description (sent to the worker; must
  /// outlive nothing — copied onto the wire).
  /// \param tier_config CPU tier sizing for the WORKER's allocation
  /// (must match the engine-side MetadataPlaneConfig id spaces).
  ///
  /// \throws stratakv::LogicError when the worker cannot start or
  /// reports a setup failure (CUDA IPC unsupported carries a stable
  /// message prefix — see IsIpcUnsupported()).
  /// \brief Spawn with default config (overload pair, not a default
  /// argument: Config's NSDMIs cannot appear in same-class default
  /// args).
  IpcTransportClient(const common::ModelLayout& layout,
                     CpuTierConfig tier_config);
  /// \brief Spawn with explicit options.
  /// \param layout Model description, copied onto the wire.
  /// \param tier_config Worker CPU tier sizing.
  /// \param config Spawn/timeout options.
  IpcTransportClient(const common::ModelLayout& layout,
                     CpuTierConfig tier_config, Config config);
  ~IpcTransportClient() override;

  IpcTransportClient(const IpcTransportClient&) = delete;
  IpcTransportClient& operator=(const IpcTransportClient&) = delete;

  /// \brief Export one GPU group's buffers to the worker (cudaIpc-
  /// GetMemHandle here; cudaIpcOpenMemHandle there). Must be called
  /// before the first Enqueue; one per group like the registry.
  ///
  /// \throws stratakv::LogicError on export/worker failure.
  void RegisterGpuGroup(const GroupDescriptor& descriptor,
                        std::uint32_t capacity_blocks);

  /// \brief Non-blocking: serializes the batch to the worker. If the
  /// worker died, every op completes with ok=false inline (exactly
  /// once per op, per contract).
  void Enqueue(std::vector<control::TransferOp> batch,
               control::CompletionSink& sink) override;

  /// \brief Ask the worker to drain, wait up to grace, SIGKILL the
  /// remainder. Idempotent; Enqueue afterwards fails all ops inline.
  void DrainPending(std::chrono::milliseconds grace) override;

  std::size_t CancelPending(common::TaskHandle owner) override {
    (void)owner;
    return 0;  // Everything handed to us starts immediately.
  }

  /// \brief Ops sent but not yet completion-reported.
  std::size_t num_in_flight() const;

  /// \brief The worker process id (fault-injection hook, E4).
  pid_t worker_pid() const;

  /// \brief True when a failure message means "CUDA IPC not supported
  /// on this platform" (e.g. WSL2) — callers translate to a skip.
  static bool IsIpcUnsupported(const std::string& message);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace stratakv::data
