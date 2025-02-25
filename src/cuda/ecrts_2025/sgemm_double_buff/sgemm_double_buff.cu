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
#define WARP_COUNT 4

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

__global__ void sgemm_shmem(int M, int N, int K, float alpha, const float *A, 
                      const float *B, float beta, float *C)
{
  // the output block that we want to compute in this threadblock
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for load block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As_1[BLOCKSIZE * BLOCKSIZE];
  __shared__ float As_2[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs_1[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs_2[BLOCKSIZE * BLOCKSIZE];
  float *load_A = As_1;
  float *load_B = Bs_1;
  float *process_A = As_2;
  float *process_B = Bs_2;

  float tmpA, tmpB;
  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  load_A[threadRow * BLOCKSIZE + threadCol] = A[threadCol * K + threadCol];
  load_B[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
  int bkIdx;
  __syncthreads();
  for (bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    
    load_A = load_A == As_1 ? As_2 : As_1;
    load_B = load_B == Bs_1 ? Bs_2 : Bs_1;
    process_A = load_A == As_1 ? As_2 : As_1;
    process_B = load_B == Bs_1 ? Bs_2 : Bs_1;
    // Advancing global mem pointers
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
    // Loading for next loop
    tmpA = A[threadRow * K + threadCol];
    tmpB = B[threadRow * N + threadCol];
    // execute the dotproduct on the loadly cached block
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += process_A[threadRow * BLOCKSIZE + dotIdx] *
            process_B[dotIdx * BLOCKSIZE + threadCol];
    }
    // block threads in this block until cache is fully populated
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    // __syncthreads();
    load_A[threadRow * BLOCKSIZE + threadCol] = tmpA;
    load_B[threadRow * BLOCKSIZE + threadCol] = tmpB;
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
    M = WARP_COUNT;
    N = WARP_COUNT;
    K = 32;
    float hA[M * K];
    float hB[N * K];
    float hC[M * N];
    

    dim3 gridDim(1, 1, 1);
    dim3 blockDim( 32, WARP_COUNT, 1);
    
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