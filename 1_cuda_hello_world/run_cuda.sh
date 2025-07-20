#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=0

nvcc -o output_cuda.out cuda_exercise.cu \
  -gencode arch=compute_50,code=sm_50
./output_cuda.out
