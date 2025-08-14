#include <chrono>
#include <cstdio>
#include <iostream>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// softmax on tensor of size [ROW, COL]
// where ROW stands for batch size, we do softmax on COL
const int ROW = 4097;
const int COL = 4093;

const int WARP_SIZE = 32;

template <typename T, typename Op>
__device__ __inline__ T warpReduce(T val, Op op,
                                   unsigned int mask = 0xffffffffu) {
  for (int offset = 16; offset > 0; offset >>= 1)
    val = op(val, __shfl_down_sync(mask, val, offset));
  return val;
}

template <typename T, typename Op>
__device__ __inline__ void blockReduce(T val, T *smem, T default_val, Op op) {
  int tx = threadIdx.x;
  int wid = tx / WARP_SIZE;
  int lane = tx % WARP_SIZE;

  val = warpReduce<T, Op>(val, op);

  if (blockDim.x > WARP_SIZE) {
    if (lane == 0) {
      smem[wid] = val;
    }
    __syncthreads();

    if (tx < WARP_SIZE) {
      val = (tx < CEIL_DIV(blockDim.x, WARP_SIZE)) ? smem[tx] : default_val;
      val = warpReduce(val, op);
      if (tx == 0) {
        smem[0] = val;
      }
    }
  } else {
    if (tx == 0) {
      smem[0] = val;
    }
  }
}

template <typename T>
__device__ __inline__ void blockReduceSum(T val, T *smem) {
  blockReduce(val, smem, static_cast<T>(0.0),
              [] __device__(T a, T b) { return a + b; });
}

template <typename T>
__device__ __inline__ void blockReduceMax(T val, T *smem) {
  blockReduce(val, smem, static_cast<T>(-INFINITY),
              [] __device__(T a, T b) { return a > b ? a : b; });
}

template <typename T, typename vecT>
__global__ void softmaxKernel(const T *__restrict__ X, T *__restrict__ Out,
                              int row, int col) {
  extern __shared__ T smem[];
  

  T local_max = -INFINITY;
  T local_norm = 0.0;

  int tid = threadIdx.x;
  int col_begin = blockIdx.x * col;
  const T *input_row = X + col_begin;
  T *output_row = Out + col_begin;
  // cuda vectorize for float4, int4 must align with a multiple of 4
  int align_head = CEIL_DIV(col_begin, 4) * 4;
  int head = align_head - col_begin;
  int col_div_4 = (col - head) / 4;
  int rem = (col - head) % 4;
  const vecT *input_row_vec = reinterpret_cast<const vecT *>(X + align_head);
  vecT *output_row_vec = reinterpret_cast<vecT *>(Out + align_head);
  T maxval = -INFINITY;

#pragma unroll
  for (int i = tid; i < col_div_4; i += blockDim.x) {
    //printf("i = %d, r = %d\n", i, blockIdx.x);
    vecT elem = input_row_vec[i];
    //printf("i = %d, elem = %f %f %f %f\n", i, elem.x, elem.y, elem.z, elem.w);

    maxval = max(maxval, elem.x);
    maxval = max(maxval, elem.y);
    maxval = max(maxval, elem.z);
    maxval = max(maxval, elem.w);
    if (maxval > local_max) {
      local_norm *= exp(local_max - maxval);
      local_max = maxval;
    }
    local_norm += exp(elem.x - maxval);
    local_norm += exp(elem.y - maxval);
    local_norm += exp(elem.z - maxval);
    local_norm += exp(elem.w - maxval);
  }
  // if (tid == 0) printf("Debug before rem block %d\n", blockIdx.x);
  if (head && tid < head) {
    T val = input_row[tid];
    if (val > local_max) {
      local_norm *= exp(local_max - val);
      local_max = val;
    }
    local_norm += exp(val - local_max);
  }
  if (rem && tid >= head && tid < head + rem) {
    T val = input_row[col_div_4 * 4 + tid];
    if (val > local_max) {
      local_norm *= exp(local_max - val);
      local_max = val;
    }
    local_norm += exp(val - local_max);
  }

  __syncthreads();

  // if (tid == 0) printf("Debug before x_max block %d\n", blockIdx.x);
  blockReduceMax<T>(local_max, smem);
  __syncthreads();

  T x_max = smem[0];
  __syncthreads();

  T val = local_norm * exp(local_max - x_max);
  blockReduceSum<T>(val, smem);
  __syncthreads();

  T e_sum = smem[0];
  __syncthreads();

  //if (tid == 0) {
  //  printf("cuda r = %d, x_max = %f\n", blockIdx.x, x_max);
  //  printf("cuda r = %d, e_sum = %f\n", blockIdx.x, e_sum);
  //}
#pragma unroll
  for (int i = tid; i < col_div_4; i += blockDim.x) {
    vecT elem = input_row_vec[i];
    elem.x = exp(elem.x - x_max) / e_sum;
    elem.y = exp(elem.y - x_max) / e_sum;
    elem.z = exp(elem.z - x_max) / e_sum;
    elem.w = exp(elem.w - x_max) / e_sum;

    output_row_vec[i] = elem;
  }
  if (head && tid < head) {
    T val = input_row[tid];
    output_row[tid] = exp(val - x_max) / e_sum;
  }
  if (rem && tid >= head && tid < head + rem) {
    int pos = col_div_4 * 4 + tid;
    T val = input_row[pos];
    output_row[pos] = exp(val - x_max) / e_sum;
  }
}

template <typename T, typename vecT>
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
  size_t smem_size = CEIL_DIV(num_threads, WARP_SIZE) * sizeof(T);
  softmaxKernel<T, vecT>
      <<<row, num_threads, smem_size>>>(dev_X, dev_Out, row, col);
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
    //printf("cpu r = %d, x_max = %f\n", r, x_max);
    //printf("cpu r = %d, e_sum = %f\n", r, e_sum);

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
    // X[i] = static_cast<float>((i) % 10);
    X[i] = dist(gen);
  }
  printf("Random generated Matrix size row = %d, col = %d\n", ROW, COL);

  float *cpu_Out = new float[M];
  cpuSoftmax<float>(X, cpu_Out, ROW, COL);

  float *cuda_Out = new float[M];
  cudaSoftmax<float, float4>(X, cuda_Out, ROW, COL);

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
