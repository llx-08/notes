#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
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

// 教学用的朴素 GEMM：
// C[M,N] = A[M,K] × B[K,N]，三者均为 row-major。
// 每个 thread 计算一个 C[row, col]，没有 shared-memory tiling，也没有显式
// Tensor Core API。
__global__ void naive_matmul(const float* a, const float* b, float* c, int m,
                             int n, int k) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || col >= n) {
    return;
  }

  float sum = 0.0f;
  for (int inner = 0; inner < k; ++inner) {
    sum += a[row * k + inner] * b[inner * n + col];
  }
  c[row * n + col] = sum;
}

int main() {
  constexpr int m = 512;
  constexpr int n = 512;
  constexpr int k = 512;
  constexpr int warmup = 3;
  constexpr int repeat = 20;

  std::mt19937 generator(7);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
  std::vector<float> host_a(static_cast<size_t>(m) * k);
  std::vector<float> host_b(static_cast<size_t>(k) * n);
  std::vector<float> host_c(static_cast<size_t>(m) * n);
  for (float& value : host_a) value = distribution(generator);
  for (float& value : host_b) value = distribution(generator);

  float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
  CUDA_CHECK(cudaMalloc(&device_a, host_a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_b, host_b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_c, host_c.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_a, host_a.data(),
                        host_a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_b, host_b.data(),
                        host_b.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);

  for (int i = 0; i < warmup; ++i) {
    naive_matmul<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    naive_matmul<<<grid, block>>>(device_a, device_b, device_c, m, n, k);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float total_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

  CUDA_CHECK(cudaMemcpy(host_c.data(), device_c,
                        host_c.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // 为避免教学程序在 CPU 上再做完整 O(MNK) reference，抽查固定坐标。
  const int sample_rows[] = {0, 1, 17, m - 1};
  const int sample_cols[] = {0, 3, 29, n - 1};
  double max_sample_error = 0.0;
  for (int row : sample_rows) {
    for (int col : sample_cols) {
      double expected = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        expected += static_cast<double>(host_a[row * k + inner]) *
                    host_b[inner * n + col];
      }
      max_sample_error =
          std::max(max_sample_error,
                   std::abs(static_cast<double>(host_c[row * n + col]) -
                            expected));
    }
  }

  const double average_seconds = (total_ms / repeat) / 1e3;
  const double operations = 2.0 * m * n * k;
  std::cout << "problem: M=" << m << ", N=" << n << ", K=" << k << '\n';
  std::cout << "launch: grid=(" << grid.x << ',' << grid.y << "), block=("
            << block.x << ',' << block.y << ")\n";
  std::cout << "average latency: " << average_seconds * 1e3 << " ms\n";
  std::cout << "throughput: " << operations / average_seconds / 1e9
            << " GFLOP/s\n";
  std::cout << "max sampled error: " << max_sample_error << '\n';

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(device_a));
  CUDA_CHECK(cudaFree(device_b));
  CUDA_CHECK(cudaFree(device_c));
  return max_sample_error < 1e-4 ? EXIT_SUCCESS : EXIT_FAILURE;
}
