// Cross-process CUDA IPC feasibility probe (Phase E4 prerequisite,
// ADR 0002): parent exports a patterned cudaMalloc buffer via
// cudaIpcGetMemHandle to a file, execs itself as a child role which
// opens it with cudaIpcOpenMemHandle, verifies the bytes, and writes
// its own pattern back; the parent then verifies the child's writes.
// Exit code / PROBE RESULT line settles platform feasibility:
//   2026-11-19 E-Local (WSL2, driver 610.88): export OK, open FAILS
//     ("invalid resource handle") — isolated mode impossible there.
//   2026-11-19 E0 host (native Linux, 3080 Ti, driver 580.76.05):
//     see benchmarks/results/ logs.
// Not a ctest entry (spawns processes; environment documentation
// tool). Build with STRATAKV_GPU; run with no arguments.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

namespace {

constexpr unsigned long long kWords = 1u << 20;  // 8 MiB of u64.

__global__ void Touch(unsigned long long* p, unsigned long long v) {
  const unsigned long long i = blockIdx.x * static_cast<unsigned long long>(
                                             blockDim.x) +
                               threadIdx.x;
  if (i < kWords) {
    p[i] = v ^ i;
  }
}

int Child(const char* handle_path, const char* result_path) {
  cudaIpcMemHandle_t handle{};
  FILE* f = std::fopen(handle_path, "rb");
  if (f == nullptr || std::fread(&handle, 1, sizeof(handle), f) !=
                          sizeof(handle)) {
    std::printf("child: read handle failed\n");
    return 2;
  }
  std::fclose(f);
  void* dev = nullptr;
  cudaError_t e =
      cudaIpcOpenMemHandle(&dev, handle, cudaIpcMemLazyEnablePeerAccess);
  std::printf("child: cudaIpcOpenMemHandle: %s\n", cudaGetErrorString(e));
  if (e != cudaSuccess) {
    std::remove(result_path);
    return 3;
  }
  unsigned long long* host = nullptr;
  cudaMallocHost(&host, kWords * 8);
  std::memcpy(host, dev, 0);  // Touch the mapping (host-side sanity).
  cudaError_t e2 =
      cudaMemcpy(host, dev, kWords * 8, cudaMemcpyDeviceToHost);
  e2 = cudaDeviceSynchronize();
  std::printf("child: readback+sync: %s\n", cudaGetErrorString(e2));
  bool ok = e2 == cudaSuccess;
  if (ok) {
    for (unsigned long long i = 0; i < kWords; ++i) {
      if (host[i] != (0xABCD0000ULL ^ i)) {
        ok = false;
        break;
      }
    }
    std::printf("child: parent pattern bytes: %s\n",
                ok ? "OK" : "CORRUPT");
  }
  Touch<<<static_cast<unsigned>(kWords / 256), 256>>>(
      static_cast<unsigned long long*>(dev), 0x11110000ULL);
  const cudaError_t e3 = cudaDeviceSynchronize();
  std::printf("child: write+sync: %s\n", cudaGetErrorString(e3));
  cudaFreeHost(host);
  if (ok && e3 == cudaSuccess) {
    FILE* r = std::fopen(result_path, "w");
    if (r != nullptr) {
      std::fputs("ok", r);
      std::fclose(r);
    }
  } else {
    std::remove(result_path);
  }
  return ok && e3 == cudaSuccess ? 0 : 5;
}

int Parent() {
  unsigned long long* dev = nullptr;
  cudaError_t e = cudaMalloc(&dev, kWords * 8);
  if (e != cudaSuccess) {
    std::printf("parent: cudaMalloc: %s\n", cudaGetErrorString(e));
    return 1;
  }
  Touch<<<static_cast<unsigned>(kWords / 256), 256>>>(dev, 0xABCD0000ULL);
  cudaDeviceSynchronize();
  cudaIpcMemHandle_t handle{};
  e = cudaIpcGetMemHandle(&handle, dev);
  std::printf("parent: cudaIpcGetMemHandle: %s\n", cudaGetErrorString(e));
  if (e != cudaSuccess) {
    return 1;
  }
  const char* handle_path = "/tmp/stratakv_ipc_probe_handle";
  const char* result_path = "/tmp/stratakv_ipc_probe_result";
  std::remove(result_path);
  FILE* f = std::fopen(handle_path, "wb");
  std::fwrite(&handle, 1, sizeof(handle), f);
  std::fclose(f);

  char self[256];
  const ssize_t n = readlink("/proc/self/exe", self, sizeof(self) - 1);
  self[n > 0 ? n : 0] = '\0';
  char* argv[] = {self, const_cast<char*>("--child"),
                  const_cast<char*>(handle_path),
                  const_cast<char*>(result_path), nullptr};
  int status = 0;
  const pid_t pid = fork();
  if (pid == 0) {
    execv(self, argv);
    _exit(127);
  }
  waitpid(pid, &status, 0);
  std::printf("parent: child exit %d\n", WEXITSTATUS(status));

  bool child_ok = false;
  FILE* r = std::fopen(result_path, "r");
  if (r != nullptr) {
    char buf[8] = {0};
    if (std::fread(buf, 1, 2, r) == 2 && std::strcmp(buf, "ok") == 0) {
      child_ok = true;
    }
    std::fclose(r);
  }
  unsigned long long* host = nullptr;
  cudaMallocHost(&host, kWords * 8);
  cudaMemcpy(host, dev, kWords * 8, cudaMemcpyDeviceToHost);
  bool parent_sees_child = true;
  for (unsigned long long i = 0; i < kWords; ++i) {
    if (host[i] != (0x11110000ULL ^ i)) {
      parent_sees_child = false;
      break;
    }
  }
  std::printf("parent: sees child's writes: %s\n",
              parent_sees_child ? "OK" : "CORRUPT");
  std::printf(
      "PROBE RESULT: cross-process CUDA IPC %s\n",
      child_ok && parent_sees_child
          ? "WORKS on this platform"
          : "DOES NOT WORK on this platform (isolated mode impossible)");
  cudaFreeHost(host);
  cudaFree(dev);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 4 && std::strcmp(argv[1], "--child") == 0) {
    return Child(argv[2], argv[3]);
  }
  return Parent();
}
