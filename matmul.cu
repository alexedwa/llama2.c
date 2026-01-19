#include <stdio.h>
#include <stdio.h>
#include <omp.h>
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>

#define N 1024
#define BILLION 1000000000
#define TILE 64
void initialise();
void matmul(float* xout, float* x, float* w, int n, int d);

__align__(64) float xout[N];
__align__(64) float x[N];
__align__(64) float w[N * N];

void matmul(float* xout, float* x, float* w, int n, int d) {
    int i;
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

__global__ void matmul_cuda(float* xout, float* x, float* w, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < d && j < n) {
        float val = 0.0f;
        for (int k = 0; k < n; k++) {
            val += w[i * n + j] * x[j];
        }

        xout[i] = val;
    }
}

__global__ void matmul_cuda_tiled(float* xout, float* x, float* w, int n, int d) {
    __shared__ float aa[TILE][TILE];
    __shared__ float bb[TILE][TILE];

    float val = 0.0f;
    int k, m;

    int row_a = TILE * blockIdx.y + threadIdx.y;
    int col_b = blockIdx.x * TILE + threadIdx.x;

    for (m = 0; m < d / TILE; m++) {
        aa[threadIdx.y][threadIdx.x] = x[N * (row_a)+(m * TILE + threadIdx.x)];
        bb[threadIdx.y][threadIdx.x] = w[N * (m * TILE + threadIdx.y) + (col_b)];

        __syncthreads(); 

        for(k = 0; k < TILE; k++) {
            val += aa[threadIdx.y][k] * bb[k][threadIdx.x];
        }

        __syncthreads();
    }
    xout[d * row_a + col_b] = val;
}

void initialise() {
    int i, j;
    for (i = 0; i < N; i++) {
        xout[i] = (float)(i % 7 + 0.01);
        x[i] = (float)(i % 7 + 0.01);

        for (j = 0; j < N; j++) {
            w[i * j] = (float)(i % 7 + 0.01);

        }
    }
}

int main() {
    double start, end;
    long long flops;
    int reruns = 100000000, i = 0; 

    initialise();
    cudaError_t cudaStatus;
   
    //dim3 dimBlock(N, N, 1);
    //dim3 dimGrid(N, N, 1);

    dim3 dimBlock(N, N, 1);
    dim3 dimGrid((N + (TILE * 2) - 1) / (TILE * 2), (N + (TILE * 2) - 1) / (TILE * 2), 1);
    
    for (int j = 0; j < 5; j++) {
        start = omp_get_wtime();
        for (i = 0; i < reruns; i++) {
            //matmul(xout, x, w, N, N);

            //matmul_cuda << <dimGrid, dimBlock >> > (xout, x, w, N, N);

            matmul_cuda_tiled << <dimGrid, dimBlock >> > (xout, x, w, N, N);
        }
        end = omp_get_wtime();

        flops = (2 * N * N);
        printf("\nTotal Run Time: %f seconds\n", (end - start));
        printf("Average Run Time: %f seconds\n", ((end - start) / reruns));
        printf("GFLOPS: %f\n", ((reruns * flops) / (end - start)) / BILLION);
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        printf("\ncuda Reset failed!");
        return -1;
    }

    return 0;
}