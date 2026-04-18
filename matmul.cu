#include <stdio.h>
#include <stdio.h>
#include <omp.h>
#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>

#define N 8192 // for others 1024, 2048, 4096, 8192
#define BILLION 1000000000
#define TILE 64
void initialise();
void matmul(float* xout, float* x, float* w, int n, int d);

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

    if (i < d) {
        float val = 0.0f;
        for (int k = 0; k < n; k++) {
            val += w[i * n + k] * x[k];
        }
        xout[i] = val;
    }
}
__global__ void matmul_cuda_rb2(float* xout, float* x, float* w, int n, int d) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (i + 1 < d) {
        float val0 = 0.0f, val1 = 0.0f;
        
        for (int k = 0; k < n; k++) {
            float x_k = x[k];
            val0 += w[(i + 0) * n + k] * x_k;
            val1 += w[(i + 1) * n + k] * x_k;


        }
        
        xout[i + 0] = val0;
        xout[i + 1] = val1;


    }
    else if (i < d) {
        // Tail handling for when d is not divisible by 8
        for (int ii = i; ii < d; ii++) {
            float val = 0.0f;
            for (int k = 0; k < n; k++) {
                val += w[ii * n + k] * x[k];
            }
            xout[ii] = val;
        }
    }
}

__global__ void matmul_cuda_rb4(float* xout, float* x, float* w, int n, int d) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (i + 3 < d) {
        float val0 = 0.0f, val1 = 0.0f, val2 = 0.0f, val3 = 0.0f;
        
        for (int k = 0; k < n; k++) {
            float x_k = x[k];
            val0 += w[(i + 0) * n + k] * x_k;
            val1 += w[(i + 1) * n + k] * x_k;
            val2 += w[(i + 2) * n + k] * x_k;
            val3 += w[(i + 3) * n + k] * x_k;

        }
        
        xout[i + 0] = val0;
        xout[i + 1] = val1;
        xout[i + 2] = val2;
        xout[i + 3] = val3;

    }
    else if (i < d) {
        // Tail handling for when d is not divisible by 8
        for (int ii = i; ii < d; ii++) {
            float val = 0.0f;
            for (int k = 0; k < n; k++) {
                val += w[ii * n + k] * x[k];
            }
            xout[ii] = val;
        }
    }
}

__global__ void matmul_cuda_rb8(float* xout, float* x, float* w, int n, int d) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    
    if (i + 7 < d) {
        float val0 = 0.0f, val1 = 0.0f, val2 = 0.0f, val3 = 0.0f, val4 = 0.0f, val5 = 0.0f, val6 = 0.0f, val7 = 0.0f;
        
        for (int k = 0; k < n; k++) {
            float x_k = x[k];
            val0 += w[(i + 0) * n + k] * x_k;
            val1 += w[(i + 1) * n + k] * x_k;
            val2 += w[(i + 2) * n + k] * x_k;
            val3 += w[(i + 3) * n + k] * x_k;
            val4 += w[(i + 4) * n + k] * x_k;
            val5 += w[(i + 5) * n + k] * x_k;
            val6 += w[(i + 6) * n + k] * x_k;
            val7 += w[(i + 7) * n + k] * x_k;
        }
        
        xout[i + 0] = val0;
        xout[i + 1] = val1;
        xout[i + 2] = val2;
        xout[i + 3] = val3;
        xout[i + 4] = val4;
        xout[i + 5] = val5;
        xout[i + 6] = val6;
        xout[i + 7] = val7;

    }
    else if (i < d) {
        // Tail handling for when d is not divisible by 8
        for (int ii = i; ii < d; ii++) {
            float val = 0.0f;
            for (int k = 0; k < n; k++) {
                val += w[ii * n + k] * x[k];
            }
            xout[ii] = val;
        }
    }
}

__global__ void matmul_cuda_tiled(float* xout, float* x, float* w, int n, int d) {
    __shared__ float x_tile[TILE];
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float val = 0.0f;
    
    for (int k_tile = 0; k_tile < n; k_tile += TILE) {
        
        x_tile[tid] = x[k_tile + tid];
        
        __syncthreads();
        
        if (i < d) {
            for (int k = 0; k < TILE; k++) {
                val += w[i * n + (k_tile + k)] * x_tile[k];
            }
        }
        
        __syncthreads();
    }
    
    if (i < d) {
        xout[i] = val;
    }
}

__global__ void matmul_sw_pipe(float* xout, float* x, float* w, int n, int d) {
    __shared__ float x_tile[TILE];
    __shared__ float x_tile_next[TILE];
    
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    int k, m;
    
    float val = 0.0f;

    x_tile[tid] = x[0 * TILE + tid];
    __syncthreads();
    
    for(m = 1; m < ((N / TILE) - 1); m+=2){
        if (i < d) {
            for (k = 0; k != TILE; k++) {
                val += w[i * n + ((m - 1) * TILE + k)] * x_tile[k];
            }
        }
        x_tile_next[tid] = x[m * TILE + tid];
        __syncthreads();

        if (i < d) {
            for (k = 0; k != TILE; k++) {
                val += w[i * n + (m * TILE + k)] * x_tile_next[k];
            }
        }
        x_tile[tid] = x[(m + 1) * TILE + tid];
        __syncthreads();
    }

    // CLEANUP
    if (i < d) {
        for (k = 0; k != TILE; k++) {
            val += w[i * n + ((m - 1) * TILE + k)] * x_tile[k];
        }
    }
    m = ((n / TILE) - 1);
    x_tile_next[tid] = x[m * TILE + tid];
    __syncthreads();
    
    if (i < d) {
        for (k = 0; k != TILE; k++) {
            val += w[i * n + (m * TILE + k)] * x_tile_next[k];
        }
    }
    __syncthreads();
    
    
    if (i < d) {
        xout[i] = val;
    }
}


void initialise(float* x, float* xout, float* w) {
    for (int i = 0; i < N; i++) {
        x[i] = (float)(i % 7 + 0.01f);
        xout[i] = 0.0f;
        
        for (int j = 0; j < N; j++) {
            w[i * N + j] = (float)((i + j) % 7 + 0.01f);
        }
    }
}

int main() {
    long long flops;
    int reruns = 1000;

    cudaError_t cudaStatus;

    float* xout = (float*)malloc(N * sizeof(float));
    float* x = (float*)malloc(N * sizeof(float));
    float* w = (float*)malloc((size_t)N * N * sizeof(float));
    initialise(xout, x, w);

    // base cuda
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    //rb
    int rb_threads = 256;
    int factor_value = 8;
    int rb_blocks = (N + rb_threads * factor_value - 1) / (rb_threads * factor_value);
    
    // lt
    int lt_threads = TILE;
    int lt_blocks = (N + lt_threads - 1) / lt_threads;

    //sw
    int sw_threads = TILE;
    int sw_blocks = (N + sw_threads - 1) / sw_threads;
    
    float *x_d, *xout_d, *w_d;
    cudaMalloc(&x_d, N * sizeof(float));
    cudaMalloc(&xout_d, N * sizeof(float));
    cudaMalloc(&w_d, N * N * sizeof(float));
    
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, w, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);
    
    //warmup
    matmul_cuda<<<blocks, threads>>>(xout_d, x_d, w_d, N, N);
    cudaDeviceSynchronize();
    
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return -1;
    }
    
    for (int j = 0; j < 5; j++) {
        cudaEventRecord(start_ev);
        
        for (int i = 0; i < reruns; i++) {
            //matmul_cuda<<<blocks, threads>>>(xout_d, x_d, w_d, N, N);

            //matmul_cuda_rb2<<<rb_blocks, rb_threads>>>(xout_d, x_d, w_d, N, N);
            //matmul_cuda_rb4<<<rb_blocks, rb_threads>>>(xout_d, x_d, w_d, N, N);
            //matmul_cuda_rb8<<<rb_blocks, rb_threads>>>(xout_d, x_d, w_d, N, N);

            //matmul_cuda_tiled<<<lt_blocks, lt_threads>>>(xout_d, x_d, w_d, N, N);

            matmul_sw_pipe<<<sw_blocks, sw_threads>>>(xout_d, x_d, w_d, N, N);
        }
        
        cudaEventRecord(stop_ev);
        cudaEventSynchronize(stop_ev);
        
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        double seconds = ms / 1000.0;
        
        flops = 2LL * N * N;
        printf("Elapsed: %f s, GFLOPS: %f\n", seconds, ((double)reruns * flops / seconds) / BILLION);
    }
    
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    
    cudaFree(x_d);
    cudaFree(xout_d);
    cudaFree(w_d);
    cudaDeviceReset();
    return 0;
}