#include <iostream>

// https://siboehm.com/articles/22/CUDA-MMM

// MIT License

// Copyright (c) 2023 Simon Boehm

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define BLOCKSIZE 32

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void sgemm_shmem(int M, int N, int K, float alpha, const float *A, 
                      const float *B, float beta, float *C)
{
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

void generate_matrix(float *mat, int size)
{
    int i;
    for(i = 0; i < size; i++)
    {
        mat[i] = (i % 100) / 3;
    }
}

int main()
{
    float *dA, *dB, *dC;
    int M, N, K;
    M = 4;
    N = 4;
    K = 32;
    float hA[M * K];
    float hB[N * K];
    float hC[M * N];
    

    dim3 gridDim(1, 1, 1);
    dim3 blockDim( 32, 4, 1);
    
    generate_matrix(hA, M * K);
    generate_matrix(hB, N * K);
    generate_matrix(hC, M * N);
    
    cudaMalloc((void **)&dA, sizeof(float) * M * K);
    cudaMalloc((void **)&dB, sizeof(float) * K * N);
    cudaMalloc((void **)&dC, sizeof(float) * M * N);

    cudaMemcpy(dA, hA, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, hC, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    sgemm_shmem<<< gridDim, blockDim >>>(M, N, K,0.5f, dA, dB, 0.5f, dC);

    cudaMemcpy(hC, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}