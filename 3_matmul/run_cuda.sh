#!/bin/bash

set -ex

export CUDA_VISIBLE_DEVICES=4

nvcc -o output_cuda.out cuda_exercise.cu
./output_cuda.out
