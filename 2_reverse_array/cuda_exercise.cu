#include <iostream>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>


const int BLOCK_SIZE = 256;

__global__ void reverse_array_block(const int* in, int* out, const int len) {
  int write_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (write_idx >= len) {
    return;
  }
  int read_idx = len - write_idx - 1;
  out[write_idx] = in[read_idx];
}

int main() {
  int N = 1111;

  int* host_a = new int[N];
  for (int i = 0; i < N; ++i) {
    host_a[i] = i;
  }

  int size = N * sizeof(int);

  int* dev_a = nullptr;
  cudaMalloc(&dev_a, size);
  int* dev_out = nullptr;
  cudaMalloc(&dev_out, size);


  cudaMemcpy(dev_a, host_a, size, cudaMemcpyHostToDevice);

  reverse_array_block<<<(N - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(dev_a, dev_out, N);

  int* out = new int[N];
  cudaMemcpy(out, dev_out, size, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    printf("%d ", out[i]);
  }
  printf("\n");

  return 0;
}
