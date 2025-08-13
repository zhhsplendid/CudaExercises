#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

// softmax on tensor of size [ROW, COL]
// where ROW stands for batch size, we do softmax on COL
const int ROW = 4096;
const int COL = 1024;

const int SHARED_SIZE = 1024;

template <typename T> __device__ T warpReduceSum(T val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T> __device__ T warpReduceMax(T val) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = max(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

template <typename T>
__global__ void softmaxKernel(const T *__restrict__ X, T *__restrict__ Out,
                              int row, int col) {
  __shared__ T smem[SHARED_SIZE];
  int col_begin = blockIdx.x * col;
  const T *input_row = X + col_begin;
  T *output_row = Out + col_begin;

  T local_max = -INFINITY;
  T local_norm = 0.0;

  int warp_size = 32;
  int tid = threadIdx.x;
  // Handle column size larger than thread num
  for (int i = tid; i < col; i += blockDim.x) {
    T elem = input_row[i];
    if (elem > local_max) {
      local_max = elem;
      local_norm *= exp(local_max - elem);
    }
    local_norm += exp(elem - local_max);
  }
  __syncthreads();

  T val = local_max;
  //printf("tid = %d, val = %f\n", tid, val);
  val = warpReduceMax<T>(val);
  //printf("tid = %d, after reduce val = %f\n", tid, val);

  if (blockDim.x > warp_size) {
    if (tid % warp_size == 0) {
      smem[tid / warp_size] = val;
    }
    __syncthreads();

    if (tid < warp_size) {
      val = tid < blockDim.x / warp_size ? smem[tid] : -INFINITY;
      val = warpReduceMax<T>(val);
      if (tid == 0) {
        smem[0] = val;
      }
    }
  } else {
    if (tid == 0) {
      smem[0] = val;
    }
  }
  __syncthreads();

  T x_max = smem[0];
  
  val = local_norm * exp(local_max - x_max);
  val = warpReduceSum(val);

  if (blockDim.x > warp_size) {
    if (tid % warp_size == 0) {
      smem[tid / warp_size] = val;
    }
    __syncthreads();

    if (tid < warp_size) {
      val = tid < blockDim.x / warp_size ? smem[tid] : 0.0f;
      val = warpReduceSum(val);
      if (tid == 0) {
        smem[0] = val;
      }
    }
  } else {
    if (tid == 0) {
      smem[0] = val;
    }
  }
  __syncthreads();

  T e_sum = smem[0];
  //if (tid == 0) {
  //  printf("cuda r = %d x_max = %f\n", blockIdx.x, x_max);
  //  printf("cuda r = %d e_sum = %f\n", blockIdx.x, e_sum);
  //}

  for (int i = tid; i < col; i += blockDim.x) {
    output_row[i] = exp(input_row[i] - x_max) / e_sum;
  }
}

template <typename T>
void cudaSoftmax(const T *__restrict__ X, T *__restrict__ Out, int row,
                 int col) {
  T *dev_X;
  T *dev_Out;
  int M = row * col;
  cudaMalloc(&dev_X, sizeof(T) * M);
  cudaMalloc(&dev_Out, sizeof(T) * M);
  cudaMemcpyAsync(dev_X, X, sizeof(T) * M, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  // each block run a row
  int num_threads = min(1024, col);
  softmaxKernel<T><<<row, num_threads>>>(dev_X, dev_Out, row, col);

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  float mill_sec;
  cudaEventElapsedTime(&mill_sec, start, end);

  printf("Cuda Softmax takes %f milliseconds\n", mill_sec);
  cudaDeviceSynchronize();
  cudaMemcpy(Out, dev_Out, sizeof(T) * M, cudaMemcpyDeviceToHost);

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(dev_X);
  cudaFree(dev_Out);
}

// Out = Softmax(X) where the length is M
template <typename T>
void cpuSoftmax(const T *__restrict__ X, T *__restrict__ Out, int row,
                int col) {
  if (row <= 0 || col <= 0) {
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int r = 0; r < row; ++r) {
    int col_begin = r * col;
    T x_max = X[col_begin];
    for (int i = 1; i < col; ++i) {
      x_max = max(X[col_begin + i], x_max);
    }
    T e_sum = 0;
    for (int i = 0; i < col; ++i) {
      Out[col_begin + i] = exp(X[col_begin + i] - x_max);
      e_sum += Out[col_begin + i];
    }
    // printf("cpu r = %d, x_max = %f\n", r, x_max);
    // printf("cpu r = %d, e_sum = %f\n", r, e_sum);

    for (int i = 0; i < col; ++i) {
      Out[col_begin + i] /= e_sum;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  printf("Cpu softmax takes %ld milliseconds\n", duration.count());
}

bool checkEqual(const float *in1, const float *in2, int row, int col) {
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      int pos = i * col + j;
      float diff = abs(in1[pos] - in2[pos]);
      // printf("diff = %f\n", diff);
      if ((in1[pos] >= 1e-5 && diff / in1[pos] > 1e-3) || diff > 1e-4) {
        printf("Answer is False! in1[%d][%d] = %f, in2[%d][%d] = %f\n", i, j,
               in1[pos], i, j, in2[pos]);
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
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
  int M = ROW * COL;
  float *X = new float[M];
  for (int i = 0; i < M; ++i) {
    X[i] = static_cast<float>((i) % 10);
    //X[i] = dist(gen);
  }
  printf("Random generated Matrix size row = %d, col = %d\n", ROW, COL);

  float *cpu_Out = new float[M];
  cpuSoftmax(X, cpu_Out, ROW, COL);

  float *cuda_Out = new float[M];
  cudaSoftmax(X, cuda_Out, ROW, COL);

  checkEqual(cpu_Out, cuda_Out, ROW, COL);

  /*
  for (int i = 0; i < 3; ++i) {
    printf("%f %f\n", cpu_Out[i], cuda_Out[i]);
  }
  */
  delete[] X;
  delete[] cpu_Out;
  delete[] cuda_Out;

  return 0;
}
