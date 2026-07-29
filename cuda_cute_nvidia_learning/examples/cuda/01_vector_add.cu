#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = (call);                                                 \
    if (error != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(error) << std::endl;                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                          \
  } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    c[index] = a[index] + b[index];
  }
}

int main() {
  constexpr int n = 1 << 24;
  constexpr int block_size = 256;
  constexpr int warmup = 10;
  constexpr int repeat = 100;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> host_a(n);
  std::vector<float> host_b(n);
  std::vector<float> host_c(n);
  for (int i = 0; i < n; ++i) {
    host_a[i] = static_cast<float>(i % 101) * 0.25f;
    host_b[i] = static_cast<float>(i % 37) * 0.5f;
  }

  float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, bytes));
  CUDA_CHECK(cudaMalloc(&device_b, bytes));
  CUDA_CHECK(cudaMalloc(&device_c, bytes));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice));

  const int grid_size = (n + block_size - 1) / block_size;
  for (int i = 0; i < warmup; ++i) {
    vector_add<<<grid_size, block_size>>>(device_a, device_b, device_c, n);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    vector_add<<<grid_size, block_size>>>(device_a, device_b, device_c, n);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes, cudaMemcpyDeviceToHost));

  double max_error = 0.0;
  for (int i = 0; i < n; ++i) {
    const double expected = static_cast<double>(host_a[i]) + host_b[i];
    max_error = std::max(max_error, std::abs(host_c[i] - expected));
  }

  const double average_seconds = (total_ms / repeat) / 1e3;
  // 每个元素读 A、读 B、写 C，共计 3 × sizeof(float)。
  const double effective_gb =
      static_cast<double>(n) * 3.0 * sizeof(float) / 1e9;
  std::cout << "N=" << n << ", grid=" << grid_size
            << ", block=" << block_size << '\n';
  std::cout << "average latency: " << average_seconds * 1e6 << " us\n";
  std::cout << "effective bandwidth: " << effective_gb / average_seconds
            << " GB/s\n";
  std::cout << "max error: " << max_error << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
  return max_error == 0.0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
