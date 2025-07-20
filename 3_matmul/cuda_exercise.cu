#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

// Matmul
// C = A * B
// A.shape = [M, K], B.shape = [K, N], C.shape = [M, N]

void cpuMatmul(const float *__restrict__ A, const float *__restrict__ B,
               float *__restrict__ C, int M, int K, int N) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  printf("Cpu matmul takes %ld milliseconds\n", duration.count());
}

__global__ void naiveMatmulKernel(const float *__restrict__ A,
                                  const float *__restrict__ B,
                                  float *__restrict__ C, int M, int K, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < M && col < N) {
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}

void naiveMatmul(const float *A, const float *B, float *C, int M, int K,
                 int N) {
  float *dev_A;
  float *dev_B;
  float *dev_C;
  cudaMalloc(&dev_A, sizeof(float) * M * K);
  cudaMalloc(&dev_B, sizeof(float) * K * N);
  cudaMalloc(&dev_C, sizeof(float) * M * N);

  cudaMemcpyAsync(dev_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(dev_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

  const int BLOCK_SIZE = 16;
  dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  naiveMatmulKernel<<<grid, block>>>(dev_A, dev_B, dev_C, M, K, N);
  cudaEventRecord(end);
  cudaMemcpyAsync(C, dev_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);

  float mill_sec;
  cudaEventElapsedTime(&mill_sec, start, end);
  printf("Naive matmul takes %f milliseconds\n", mill_sec);
}

// Each block calculates BLOCK_TILE_M * BLOCK_TILE_N elements
const int BLOCK_TILE_M = 128;
const int BLOCK_TILE_N = 128;
const int BLOCK_TILE_K = 16;
// Each thread calculates THREAD_TILE_M * THREAD_TILE_N elements
const int THREAD_TILE_M = 8;
const int THREAD_TILE_N = 8;
// Call with block Dim3(x, y) where:
// x = (BLOCK_TILE_N + THREAD_TILE_N - 1) / THREAD_TILE_N
// y = (BLOCK_TILE_M + THREAD_TILE_N - 1) / THREAD_TILE_M
__global__ void subMtxKernel(const float *A, const float *B, float *C,
                             const int M, const int K, const int N) {
  // Opt1: Tile on Matrix, each Block/Thread computes a sub-matrix
  // The naive matmul needs 2MNK Global Memory Loads, which is much slower
  // We could make part of sub-matrix in shared memory, then Global Memory
  // Access will be M / BLOCK_TILE_M * N / BLOCK_TILE_N * K / BLOCK_TILE_K
  //	(BLOCK_TILE_M * BLOCK_TILE_K + BLOCK_TILE_N * BLOCK_TILE_K)
  // = (1 / BLOCK_TILE_M + 1 / BLOCK_TILE_N) MNK
  //
  // The BLOCK_TILE_M will be limited by shared memory size.
  // BLOCK_TILE_K * (BLOCK_TILE_M * BLOCK_TILE_N) * data_type_size <= shared mem

  __shared__ float shared_A[BLOCK_TILE_M][BLOCK_TILE_K];
  __shared__ float shared_B[BLOCK_TILE_K][BLOCK_TILE_N];

  float result_C[THREAD_TILE_M][THREAD_TILE_N] = {0.0};

  int bk_limit = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

  int block_index_m = blockIdx.y * BLOCK_TILE_M;
  int block_index_n = blockIdx.x * BLOCK_TILE_N;

  int thread_offset_m =
      (BLOCK_TILE_M + blockDim.y - 1) / blockDim.y * threadIdx.y;
  int thread_end_m =
      (BLOCK_TILE_M + blockDim.y - 1) / blockDim.y * (threadIdx.y + 1);
  int thread_offset_n =
      (BLOCK_TILE_N + blockDim.x - 1) / blockDim.x * threadIdx.x;
  int thread_end_n =
      (BLOCK_TILE_N + blockDim.x - 1) / blockDim.x * (threadIdx.x + 1);

  int thread_offset_ak =
      (BLOCK_TILE_K + blockDim.x - 1) / blockDim.x * threadIdx.x;
  int thread_end_ak =
      (BLOCK_TILE_K + blockDim.x - 1) / blockDim.x * (threadIdx.x + 1);
  int thread_offset_bk =
      (BLOCK_TILE_K + blockDim.y - 1) / blockDim.y * threadIdx.y;
  int thread_end_bk =
      (BLOCK_TILE_K + blockDim.y - 1) / blockDim.y * (threadIdx.y + 1);

  for (int bk = 0; bk < bk_limit; ++bk) {
    int block_index_k = bk * BLOCK_TILE_K;
    // Cache read of shared_A
    for (int i = thread_offset_m; i < thread_end_m; ++i) {
      for (int j = thread_offset_ak; j < thread_end_ak; ++j) {
        int index_m = block_index_m + i;
        int index_k = block_index_k + j;
        if (i < BLOCK_TILE_M && j < BLOCK_TILE_K) {
          if (index_m < M && index_k < K) {
            shared_A[i][j] = A[index_m * K + index_k];
          } else {
            shared_A[i][j] = 0;
          }
        }
      }
    }
    // Cache read of shared_B
    for (int i = thread_offset_bk; i < thread_end_bk; ++i) {
      for (int j = thread_offset_n; j < thread_end_n; ++j) {
        int index_k = block_index_k + i;
        int index_n = block_index_n + j;
        if (i < BLOCK_TILE_K && j < BLOCK_TILE_N) {
          if (index_k < K && index_n < N) {
            shared_B[i][j] = B[index_k * N + index_n];
          } else {
            shared_B[i][j] = 0;
          }
        }
      }
    }
    __syncthreads();
    for (int k = 0; k < BLOCK_TILE_K; ++k) {
      for (int m = 0; m < THREAD_TILE_M; ++m) {
        for (int n = 0; n < THREAD_TILE_N; ++n) {
          int shared_index_m = threadIdx.y * THREAD_TILE_M + m;
          int shared_index_n = threadIdx.x * THREAD_TILE_N + n;

          result_C[m][n] +=
              shared_A[shared_index_m][k] * shared_B[k][shared_index_n];
	  
	  // Debug code
	  /*
          if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("result_C[%d][%d] = %f, shared_A[%d][%d] = %f, "
                   "shared_B[%d][%d] = %f\n",
                   m, n, result_C[m][n],
		   shared_index_m, k, shared_A[shared_index_m][k],
		   k, shared_index_n, shared_B[k][shared_index_n]);
          }
	  */
        }
      }
    }
    __syncthreads();
  }
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    int out_index_m =
        blockIdx.y * BLOCK_TILE_M + threadIdx.y * THREAD_TILE_M + i;
    for (int j = 0; j < THREAD_TILE_N; ++j) {
      int out_index_n =
          blockIdx.x * BLOCK_TILE_N + threadIdx.x * THREAD_TILE_N + j;
      if (out_index_m < M && out_index_n < N) {
        C[out_index_m * N + out_index_n] = result_C[i][j];
      }
    }
  }
}

// Call with block Dim3(x, y) where:
// x = (BLOCK_TILE_N + THREAD_TILE_N - 1) / THREAD_TILE_N
// y = (BLOCK_TILE_M + THREAD_TILE_N - 1) / THREAD_TILE_M
__global__ void subMtxMinorOptKernel(const float *A, const float *B, float *C,
                             const int M, const int K, const int N) {
  // Opt2: compared to tile on matrix version, this minor opt adds some unroll,
  // Change shared_A[BLOCK_TILE_M][BLOCK_TILE_K] to shared_A[BLOCK_TILE_K][BLOCK_TILE_M]
  // to avoid bank conflict.
  //
  // Also add some #pragma unroll
  __shared__ float shared_A[BLOCK_TILE_K][BLOCK_TILE_M];
  __shared__ float shared_B[BLOCK_TILE_K][BLOCK_TILE_N];

  float result_C[THREAD_TILE_M][THREAD_TILE_N] = {0.0};

  int bk_limit = (K + BLOCK_TILE_K - 1) / BLOCK_TILE_K;

  int block_index_m = blockIdx.y * BLOCK_TILE_M;
  int block_index_n = blockIdx.x * BLOCK_TILE_N;

  int thread_offset_m =
      (BLOCK_TILE_M + blockDim.y - 1) / blockDim.y * threadIdx.y;
  int thread_end_m =
      (BLOCK_TILE_M + blockDim.y - 1) / blockDim.y * (threadIdx.y + 1);
  int thread_offset_n =
      (BLOCK_TILE_N + blockDim.x - 1) / blockDim.x * threadIdx.x;
  int thread_end_n =
      (BLOCK_TILE_N + blockDim.x - 1) / blockDim.x * (threadIdx.x + 1);

  int thread_offset_ak =
      (BLOCK_TILE_K + blockDim.x - 1) / blockDim.x * threadIdx.x;
  int thread_end_ak =
      (BLOCK_TILE_K + blockDim.x - 1) / blockDim.x * (threadIdx.x + 1);
  int thread_offset_bk =
      (BLOCK_TILE_K + blockDim.y - 1) / blockDim.y * threadIdx.y;
  int thread_end_bk =
      (BLOCK_TILE_K + blockDim.y - 1) / blockDim.y * (threadIdx.y + 1);

  for (int bk = 0; bk < bk_limit; ++bk) {
    int block_index_k = bk * BLOCK_TILE_K;
    // Cache read of shared_A
    for (int i = thread_offset_m; i < thread_end_m; ++i) {
      for (int j = thread_offset_ak; j < thread_end_ak; ++j) {
        int index_m = block_index_m + i;
        int index_k = block_index_k + j;
        if (i < BLOCK_TILE_M && j < BLOCK_TILE_K) {
          if (index_m < M && index_k < K) {
            shared_A[j][i] = A[index_m * K + index_k];
          } else {
            shared_A[j][i] = 0;
          }
        }
      }
    }
    // Cache read of shared_B
    for (int i = thread_offset_bk; i < thread_end_bk; ++i) {
      for (int j = thread_offset_n; j < thread_end_n; ++j) {
        int index_k = block_index_k + i;
        int index_n = block_index_n + j;
        if (i < BLOCK_TILE_K && j < BLOCK_TILE_N) {
          if (index_k < K && index_n < N) {
            shared_B[i][j] = B[index_k * N + index_n];
          } else {
            shared_B[i][j] = 0;
          }
        }
      }
    }
    __syncthreads();
    //#pragma unroll
    for (int k = 0; k < BLOCK_TILE_K; ++k) {
      //#pragma unroll
      for (int m = 0; m < THREAD_TILE_M; ++m) {
        //#pragma unroll
	for (int n = 0; n < THREAD_TILE_N; ++n) {
          int shared_index_m = threadIdx.y * THREAD_TILE_M + m;
          int shared_index_n = threadIdx.x * THREAD_TILE_N + n;

          result_C[m][n] +=
              shared_A[k][shared_index_m] * shared_B[k][shared_index_n];

          // Debug code
          /*
          if (threadIdx.x == 0 && threadIdx.y == 0) {
            printf("result_C[%d][%d] = %f, shared_A[%d][%d] = %f, "
                   "shared_B[%d][%d] = %f\n",
                   m, n, result_C[m][n],
                   k, shared_index_m, shared_A[k][shared_index_m],
                   k, shared_index_n, shared_B[k][shared_index_n]);
          }
          */
        }
      }
    }
    __syncthreads();
  }
  //#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    int out_index_m =
        blockIdx.y * BLOCK_TILE_M + threadIdx.y * THREAD_TILE_M + i;
    //#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; ++j) {
      int out_index_n =
          blockIdx.x * BLOCK_TILE_N + threadIdx.x * THREAD_TILE_N + j;
      if (out_index_m < M && out_index_n < N) {
        C[out_index_m * N + out_index_n] = result_C[i][j];
      }
    }
  }
}

//TODO: adding matmul with 2 4 * 4 sub-matrix in a thread to avoid bank conflict
//TODO: add vectorization
//TODO: add dual buffering, that is, when do load1 - sync - compute1 - sync - load2 - sync - compute2 -sync ... loop
// we could double load buffer sizes, and do load1 - syn - load2 - compute1 - sync - load3 - compute2 -sync ... loop
// this is like pipeline parallelism which speed up load and compute. The cost is that load uses double memory.
// In matmul, it indicates double shared memory
//
// Reference: https://zhuanlan.zhihu.com/p/518857175
// based on above technic, we can get cublas speed.
//


void subMtxMatmul(const float *A, const float *B, float *C, int M, int K,
                  int N, const std::string& kernel_option) {
  float *dev_A;
  float *dev_B;
  float *dev_C;
  cudaMalloc(&dev_A, sizeof(float) * M * K);
  cudaMalloc(&dev_B, sizeof(float) * K * N);
  cudaMalloc(&dev_C, sizeof(float) * M * N);

  cudaMemcpyAsync(dev_A, A, sizeof(float) * M * K, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(dev_B, B, sizeof(float) * K * N, cudaMemcpyHostToDevice);

  dim3 grid((N + BLOCK_TILE_N - 1) / BLOCK_TILE_N,
            (M + BLOCK_TILE_M - 1) / BLOCK_TILE_M);

  dim3 block((BLOCK_TILE_N + THREAD_TILE_N - 1) / THREAD_TILE_N,
             (BLOCK_TILE_M + THREAD_TILE_M - 1) / THREAD_TILE_M);

  //printf("grid.x = %d, grid.y = %d\n", grid.x, grid.y);
  //printf("block.x = %d, block.y = %d\n", block.x, block.y);
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);
  if (kernel_option == "subMtx") {
      subMtxKernel<<<grid, block>>>(dev_A, dev_B, dev_C, M, K, N);
  } else if (kernel_option == "subMtxMinorOpt") {
      subMtxMinorOptKernel<<<grid, block>>>(dev_A, dev_B, dev_C, M, K, N);
  }
  cudaEventRecord(end);
  cudaMemcpyAsync(C, dev_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);

  float mill_sec;
  cudaEventElapsedTime(&mill_sec, start, end);
  printf("Opt matmul %s takes %f milliseconds\n", kernel_option.c_str(), mill_sec);
}

bool checkEqual(const float *in1, const float *in2, int M, int N) {
  const float ERR = 1e-3;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      int index = i * N + j;
      float diff = abs(in1[index] - in2[index]);
      if (in1[index] != 0 && diff / abs(in1[index]) > ERR && diff > ERR) {
        printf("Answer is False! in1[%d][%d] = %f, in2[%d][%d] = %f\n", i, j,
               in1[index], i, j, in2[index]);
        return false;
      }
    }
  }
  printf("Answer is True.\n");
  return true;
}


int main() {
  std::srand(0);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
  const int M = 1024;
  const int K = 64;
  const int N = 1024;
  float *A = new float[M * K];
  float *B = new float[K * N];
  for (int i = 0; i < M * K; ++i) {
    A[i] = static_cast<float>(i); // dist(gen);
  }
  for (int j = 0; j < K * N; ++j) {
    B[j] = static_cast<float>(j); // dist(gen);
  }
  printf("Random generated Matrices M = %d, K = %d, N = %d\n", M, K, N);

  float *cpu_C = new float[M * N];
  cpuMatmul(A, B, cpu_C, M, K, N);

  float *naive_C = new float[M * N];
  naiveMatmul(A, B, naive_C, M, K, N);
  checkEqual(cpu_C, naive_C, M, N);

  float *opt_C = new float[M * N];
  subMtxMatmul(A, B, opt_C, M, K, N, "subMtx");
  checkEqual(naive_C, opt_C, M, N);

  subMtxMatmul(A, B, opt_C, M, K, N, "subMtxMinorOpt");
  checkEqual(naive_C, opt_C, M, N);

  delete[] A;
  delete[] B;
  delete[] cpu_C;
  delete[] naive_C;
  delete[] opt_C;
  return 0;
}
