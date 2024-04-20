#include <cmath>
#include <cstdio>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void cuda_hello_world() {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Hello world from GPU %d\n", idx);
}

int main() {
  cuda_hello_world<<<10, 10>>>();
  cudaDeviceSynchronize();
}
