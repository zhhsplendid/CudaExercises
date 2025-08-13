#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

// softmax on tensor of size [ROW, COL]
// where ROW stands for batch size, we do softmax on COL 
const int ROW = 24;
const int COL = 256;

const int SHARED_SIZE = 1024;
const int NUM_THREADS = 256;

const int OP_REDUCE_SUM = 0;
const int OP_REDUCE_MAX = 1;
// OpCode = 0 -> ReduceSum
// otherwise -> ReduceMax
// Store result in values[0], tid must be greater than len
template <typename T, int OpCode>
__device__ void warpReduce(T *values, int len) {
  int tid = threadIdx.x;
  int warp_size = 32;
  for (int step = 1; step < len; step *= warp_size) {
    if (tid * step < len) {
      unsigned mask = __ballot_sync(0xffffffff, tid * step < len);
      T val = values[tid];
      for (int offset = warp_size >> 1; offset > 0; offset >>= 1) {
        val = OpCode == OP_REDUCE_SUM ?
              val + __shfl_down_sync(mask, val, offset, warp_size):
              max(val, __shfl_down_sync(mask, val, offset, warp_size)); 
      }
      __syncwarp();
      int next_step = step * warp_size;
      if (tid % next_step == 0) {
        values[tid / next_step] = val;
      }
      __syncwarp();
    }
  }
}


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
    if (tid == 0) {
      printf("r = %d, smem[0] = %f\n", blockIdx.x, smem[0]);
    }
    __syncthreads();
    ans += smem[0];
  }
  return ans;
}

template <typename T>
__global__ void softmaxKernel(const T *__restrict__ X, T *__restrict__ Out,
                              int row, int col) {
  int col_begin = blockIdx.x * col;
  T x_max = blockReduce<T, OP_REDUCE_MAX>(X + col_begin, col);
  int tid = threadIdx.x;
  int pos = col_begin + tid;

  if (tid < col) {
    if (tid == 0) {
      printf("cuda r = %d x_max = %f\n", blockIdx.x, x_max);
    }
    Out[pos] = exp(X[pos] - x_max);
    __syncthreads();
    //printf("tid = %d, Out[pos] = %f\n", tid, Out[pos]);
    T e_sum = blockReduce<T, OP_REDUCE_SUM>(Out + col_begin, col);
    if (tid == 0) {
      printf("cuda r = %d e_sum = %f\n", blockIdx.x, e_sum);
    }
    __syncthreads();

    Out[pos] = Out[pos] / e_sum;
  }
}

template <typename T>
void cudaSoftmax(const T *__restrict__ X, T *__restrict__ Out, int row, int col) {
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
  softmaxKernel<T><<<row, NUM_THREADS>>>(dev_X, dev_Out, row, col);

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
void cpuSoftmax(const T *__restrict__ X, T *__restrict__ Out, int row, int col) {
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
    printf("cpu r = %d, x_max = %f\n", r, x_max);
    T e_sum = 0;
    for (int i = 0; i < col; ++i) {
      Out[col_begin + i] = exp(X[col_begin + i] - x_max);
      e_sum += Out[col_begin + i];
    }
    printf("cpu r = %d, e_sum = %f\n", r, e_sum);

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
      //printf("diff = %f\n", diff);
      if ((in1[pos] >= 1e-5 && diff / in1[pos] > 1e-2) || diff > 1e-4) {
        printf("Answer is False! in1[%d][%d] = %f, in2[%d][%d] = %f\n",
               i, j, in1[pos], i, j, in2[pos]);
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
    X[i] = static_cast<float>((M - i) % 10);
    //X[i] = dist(gen);
  }
  printf("Random generated Matrix size row = %d, col = %d\n", ROW, COL);

  float *cpu_Out = new float[M];
  cpuSoftmax(X, cpu_Out, ROW, COL);

  float *cuda_Out = new float[M];
  cudaSoftmax(X, cuda_Out, ROW, COL);
  
  checkEqual(cpu_Out, cuda_Out, ROW, COL);

  for (int i = 0; i < 3; ++i) {
    printf("%f %f\n", cpu_Out[i], cuda_Out[i]);
  }
  delete[] cpu_Out;
  delete[] cuda_Out;

  return 0;
}
