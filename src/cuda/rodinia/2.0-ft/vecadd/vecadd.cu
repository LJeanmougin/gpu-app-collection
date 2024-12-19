#include <stdio.h>

#define BSIZE 32
#define NUMBLOCK 1

__global__ void vecaddKernel(int *v1_in, int *v2_in, int *v3_in, int *v_out, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size)
        v_out[idx] = v1_in[idx] + v2_in[idx] + v3_in[idx];
        // v_out[idx] = v1_in[idx] + v2_in[idx];
        // v_out[idx] = v1_in[idx] + idx;
}

int main()
{
    int h_vec1[NUMBLOCK * BSIZE];
    int h_vec2[NUMBLOCK * BSIZE];
    int h_vec3[NUMBLOCK * BSIZE];
    int h_out[NUMBLOCK * BSIZE];
    int *d_vec1, *d_vec2, *d_vec3, *d_out;
    for(int i = 0; i < NUMBLOCK * BSIZE; i++)
    {
        h_vec1[i] = i/3;
        h_vec2[i] = i/3;
        h_vec3[i] = i/3;
    }
    cudaMalloc((void **) &d_vec1, NUMBLOCK * BSIZE * sizeof(int));
    cudaMalloc((void **) &d_vec2, NUMBLOCK * BSIZE * sizeof(int));
    cudaMalloc((void **) &d_vec3, NUMBLOCK * BSIZE * sizeof(int));
    cudaMalloc((void **) &d_out, NUMBLOCK * BSIZE * sizeof(int));
    cudaMemcpy(d_vec1, h_vec1, NUMBLOCK * BSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, h_vec2, NUMBLOCK * BSIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec3, h_vec3, NUMBLOCK * BSIZE * sizeof(int), cudaMemcpyHostToDevice);
    vecaddKernel<<<NUMBLOCK, BSIZE>>>(d_vec1, d_vec2, d_vec3, d_out, NUMBLOCK * BSIZE);
    cudaMemcpy(h_out, d_out, NUMBLOCK * BSIZE * sizeof(int), cudaMemcpyDeviceToHost);
    return 0;
}