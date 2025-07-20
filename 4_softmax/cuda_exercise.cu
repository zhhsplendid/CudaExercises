#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

const int M = 512;

// Out = Softmax(X) where the length is M
template <typename T>
void cpuSoftmax(const T *__restrict__ X, T *__restrict__ Out, int M) {
  if (M <= 0) {
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();
  T x_max = X[0];
  for (int i = 1; i < M; ++i) {
    x_max = max(X[i], x_max);
  }
  //printf("cpu x_max = %f\n", x_max);

  T e_sum = 0;
  for (int i = 0; i < M; ++i) {
    Out[i] = exp(X[i] - x_max);
    e_sum += Out[i];
  }
  //printf("cpu e_sum = %f\n", e_sum);

  for (int i = 0; i < M; ++i) {
    Out[i] /= e_sum;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  printf("Cpu softmax takes %ld milliseconds\n", duration.count());
}

const int OP_REDUCE_SUM = 0;
const int OP_REDUCE_MAX = 1;
// OpCode = 0 -> ReduceSum
// OpCode = 1 -> ReduceMax
// Store result in values[0], tid must be greater than len
template <typename T, int OpCode>
__device__ void warpReduce(T *values, int len) {

  int tid = threadIdx.x;
  int warp_size = 32;
  for (int step = 1; step < len; step *= warp_size) {
    if (tid * step < len) {
      unsigned mask = __ballot_sync(0xffffffff, tid * step < len);
      T val = values[tid];
      // 32 == group == warp size
      for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        if (OpCode == OP_REDUCE_SUM) {
          val += __shfl_down_sync(mask, val, offset, warp_size);
          //printf("tid = %d, val = %f\n", tid, val);
        } else if (OpCode == OP_REDUCE_MAX) {
          val = max(val, __shfl_down_sync(mask, val, offset, warp_size));
        }
      }
      __syncwarp();
      int next_step = step * warp_size;
      if (tid % next_step == 0) {
        values[tid / next_step] = val;
      }
    }
  }
}

const int SHARED_SIZE = M;

template <typename T, int OpCode>
__device__ T blockReduce(const T *values, int len) {
  __shared__ T smem[SHARED_SIZE];
  int NUM = (len + SHARED_SIZE - 1) / SHARED_SIZE;
  T ans = 0;
  int tid = threadIdx.x;
  for (int i = 0; i < NUM; ++i) {
    if (i * SHARED_SIZE + tid < len && tid < SHARED_SIZE) {
      smem[tid] = values[i * SHARED_SIZE + tid];
    }
    int rem_len = min((len - i * SHARED_SIZE), SHARED_SIZE);
      
    __syncthreads();
    warpReduce<T, OpCode>(smem, rem_len); 
    //printf("tid = %d, smem[0] = %f\n", tid, smem[0]);
    __syncthreads();
    ans += smem[0];
  }
  return ans;
}

template <typename T>
__global__ void softmaxKernel(const T *__restrict__ X, T *__restrict__ Out,
                              int M) {
  T x_max = blockReduce<T, OP_REDUCE_MAX>(X, M);
  int tid = threadIdx.x;
  if (tid < M) {
    if (tid == 0) {
      //printf("cuda x_max = %f\n", x_max);
    }
    Out[tid] = exp(X[tid] - x_max);
    __syncthreads();
    //printf("tid = %d, Out[tid] = %f\n", tid, Out[tid]);
    T e_sum = blockReduce<T, OP_REDUCE_SUM>(Out, M);
    if (tid == 0) {
      //printf("cuda e_sum = %f\n", e_sum);
    }
    __syncthreads();

    Out[tid] = Out[tid] / e_sum;
  }
}

template <typename T>
void cudaSoftmax(const T *__restrict__ X, T *__restrict__ Out, int M) {
  T *dev_X;
  T *dev_Out;

  cudaMalloc(&dev_X, sizeof(T) * M);
  cudaMalloc(&dev_Out, sizeof(T) * M);
  cudaMemcpyAsync(dev_X, X, sizeof(T) * M, cudaMemcpyHostToDevice);

  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  softmaxKernel<T><<<1, M>>>(dev_X, dev_Out, M);

  cudaEventRecord(end);
  cudaMemcpyAsync(Out, dev_Out, sizeof(T) * M, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  cudaFree(dev_X);
  cudaFree(dev_Out);

  float mill_sec;
  cudaEventElapsedTime(&mill_sec, start, end);
  printf("Cuda Softmax takes %f milliseconds\n", mill_sec);
}

bool checkEqual(const float *in1, const float *in2, int M) {
  const float ERR = 1e-3;
  for (int i = 0; i < M; ++i) {
    float diff = abs(in1[M] - in2[M]);
    if (in1[M] != 0 && diff / abs(in1[M]) > ERR && diff > ERR) {
      printf("Answer is False! in1[%d] = %f, in2[%d] = %f\n", M, in1[M], M,
             in2[M]);
      return false;
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
  float *X = new float[M];
  for (int i = 0; i < M; ++i) {
    //X[i] = static_cast<float>(M - i);
    X[i] = dist(gen);
  }
  printf("Random generated Matrices M = %d\n", M);

  float *cpu_Out = new float[M];
  cpuSoftmax(X, cpu_Out, M);

  float *cuda_Out = new float[M];
  cudaSoftmax(X, cuda_Out, M);
  
  checkEqual(cpu_Out, cuda_Out, M);

  for (int i = 0; i < 3; ++i) {
    printf("%f %f\n", cpu_Out[i], cuda_Out[i]);
  }
  delete[] cpu_Out;
  delete[] cuda_Out;

  return 0;
}
