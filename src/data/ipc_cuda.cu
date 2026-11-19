// CudaIpcHandleSource (Phase E1): real cudaIpcGetMemHandle behind the
// registry's IIpcHandleSource seam. The registry's contract (ADR 0005)
// keeps the 64-byte payload pristine; validity is registry-side state.

#include <cstring>
#include <string>

#include "stratakv/common/exceptions.h"
#include "stratakv/data/transport_cuda.h"

namespace stratakv::data {

IpcHandle CudaIpcHandleSource::Export(const common::BufferView& view) {
  static_assert(sizeof(cudaIpcMemHandle_t) == sizeof(IpcHandle::opaque),
                "CUDA documents the IPC handle as a 64-byte opaque blob");
  cudaIpcMemHandle_t handle{};
  cudaError_t err = cudaIpcGetMemHandle(&handle, view.base);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();  // Non-sticky query error; clear anyway.
    throw LogicError(
        std::string("CudaIpcHandleSource: cudaIpcGetMemHandle: ") +
        cudaGetErrorString(err) +
        " (CUDA IPC requires exact cudaMalloc bases and is unsupported "
        "on some platforms — WSL2 among them)");
  }
  IpcHandle out;
  std::memcpy(out.opaque.data(), &handle, sizeof(handle));
  return out;
}

}  // namespace stratakv::data
