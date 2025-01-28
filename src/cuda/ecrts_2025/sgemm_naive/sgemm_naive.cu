#include <stdio.h>
#include <math.h>

#define BSIZE 1024

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, 
                            const float *B, float beta, float *C)
{
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N)
    {
        float tmp = 0.0f;
        for(int i = 0; i < K; i++)
            tmp += A[x * K + i] * B[i * N + y];
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main()
{
    float h_mat_A[BSIZE];
    float h_mat_B[BSIZE];
    float h_mat_C[BSIZE];
    float *d_mat_A, *d_mat_B, *d_mat_C;
    
    for (int i = 0; i < BSIZE; i++)
    {
        h_mat_A[i] = i / 1000.0f;
        h_mat_B[i] = i / 1000.0f;
        h_mat_C[i] = i / 1000.0f;
    }

    dim3 gridDim(1, 1, 1);
    dim3 blockDim(32, 32, 1);

    cudaMalloc((void **) &d_mat_A, BSIZE * sizeof(float));
    cudaMalloc((void **) &d_mat_B, BSIZE * sizeof(float));
    cudaMalloc((void **) &d_mat_C, BSIZE * sizeof(float));

    cudaMemcpy(d_mat_A, h_mat_A, BSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_B, h_mat_B, BSIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat_C, h_mat_C, BSIZE * sizeof(float), cudaMemcpyHostToDevice);

    sgemm_naive<<<gridDim, blockDim>>>(32, 32, 32, 0.5, d_mat_A, d_mat_B, 0.5, d_mat_C);

    cudaMemcpy(h_mat_C, d_mat_C, BSIZE * sizeof(float), cudaMemcpyDeviceToHost);

    return 0;
}