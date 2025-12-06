#include <stdio.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>

#define N 4096
#define BILLION 1000000000

void matmul(float* xout, float* x, float* w, int n, int d);
void matmul_vectorised(float* xout, float* x, float* w, int n, int d);

// Register blocking 
void matmul_rb_2(float* xout, float* x, float* w, int n, int d);
void matmul_rb_4(float* xout, float* x, float* w, int n, int d);
void matmul_rb_8(float* xout, float* x, float* w, int n, int d);

//Register blocking (VEC)
void matmul_vec_rb_2(float* xout, float* x, float* w, int n, int d);
void matmul_vec_rb_4(float* xout, float* x, float* w, int n, int d);
void matmul_vec_rb_8(float* xout, float* x, float* w, int n, int d);

// Loop tiling
void matmul_loop_tiling(float* xout, float* x, float* w, int n, int d, int TILE);
void matmul_vec_loop_tiling(float* xout, float* x, float* w, int n, int d, int TILE);


void initialise();

float xout[N] __attribute__((aligned(64)));
float x[N] __attribute__((aligned(64)));
float w[N * N] __attribute__((aligned(64)));

void matmul(float* xout, float* x, float* w, int n, int d){
    int i;
    for (i = 0; i < d; i++){
        float val = 0.0f;
        for(int j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmul_vectorised(float* xout, float* x, float* w, int n, int d){
    __m256 num1, num2, num3, num4;
    __m128 xmm1;

    int i;
    for (i = 0; i < d; i++){
        //float val = 0.0f;
        
        num1 = _mm256_setzero_ps();

        for(int j = 0; j < n; j += 8){
            //val += w[i * i + j] * x[j];

            num2 = _mm256_loadu_ps(&w[i * n + j]);
            num3 = _mm256_loadu_ps(&x[j]);
            num1 = _mm256_fmadd_ps(num2, num3, num1);
        }
        //xout[i] = val;

        num4 = _mm256_permute2f128_ps(num1, num1, 1);
        num1 = _mm256_add_ps(num1, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);

        _mm_store_ss(&xout[i], xmm1);
    }
}

// Register Blocking

void matmul_rb_2(float* xout, float* x, float* w, int n, int d){
    float val_x;

    for (int i = 0; i < d; i += 2) {
        float val1 = 0.0f;
        float val2 = 0.0f;
        
        for (int j = 0; j < n; j++) {
            val_x = x[j];

            val1 += w[i * n + j] * val_x;
            val2 += w[(i + 1) * n + j] * val_x;
        }
        
        xout[i] = val1;
        xout[i + 1] = val2;        
    }
}

void matmul_rb_4(float* xout, float* x, float* w, int n, int d) {
    float val_x;

    for (int i = 0; i < d; i += 4) {
        float val1 = 0.0f;
        float val2 = 0.0f;
        float val3 = 0.0f;
        float val4 = 0.0f;
        
        for (int j = 0; j < n; j++) {
            val_x = x[j];

            val1 += w[i * n + j] * val_x;
            val2 += w[(i + 1) * n + j] * val_x;

            val3 += w[(i + 2) * n + j] * val_x;
            val4 += w[(i + 3) * n + j] * val_x;
        }
        
        xout[i] = val1;
        xout[i + 1] = val2;
        xout[i + 2] = val3;
        xout[i + 3] = val4;        
    }
}

void matmul_rb_8(float* xout, float* x, float* w, int n, int d) {
    float val_x;
    for (int i = 0; i < d; i += 8) {
        float val1 = 0.0f;
        float val2 = 0.0f;
        float val3 = 0.0f;
        float val4 = 0.0f;
        float val5 = 0.0f;
        float val6 = 0.0f;
        float val7 = 0.0f;
        float val8 = 0.0f;
        
        for (int j = 0; j < n; j++) {
            val_x = x[j];

            val1 += w[i * n + j] * val_x;
            val2 += w[(i + 1) * n + j] * val_x;

            val3 += w[(i + 2) * n + j] * val_x;
            val4 += w[(i + 3) * n + j] * val_x;

            val5 += w[(i + 4) * n + j] * val_x;
            val6 += w[(i + 5) * n + j] * val_x;

            val7 += w[(i + 6) * n + j] * val_x;
            val8 += w[(i + 7) * n + j] * val_x; 
        }
        
        xout[i] = val1;
        xout[i + 1] = val2;
        xout[i + 2] = val3;
        xout[i + 3] = val4;

        xout[i + 4] = val5;
        xout[i + 5] = val6;
        xout[i + 6] = val7;
        xout[i + 7] = val8;
        
    }
}

// Register Blocking Vectorised
void matmul_vec_rb_2(float* xout, float* x, float* w, int n, int d) {
    __m256 num1, num2, num3, num4, num5, num6;
    __m128 xmm1;
    
    for (int i = 0; i < d; i += 2) {
        num5 = _mm256_setzero_ps();
        num6 = _mm256_setzero_ps();

        
        for (int j = 0; j < n; j += 8) {
            num3 = _mm256_loadu_ps(&x[j]);
            
            num2 = _mm256_loadu_ps(&w[i * n + j]);
            num5 = _mm256_fmadd_ps(num2, num3, num5);
            
            num2 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
            num6 = _mm256_fmadd_ps(num2, num3, num6);
          
        }
        
        num4 = _mm256_permute2f128_ps(num5, num5, 1);
        num1 = _mm256_add_ps(num5, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i], xmm1);
        
        num4 = _mm256_permute2f128_ps(num6, num6, 1);
        num1 = _mm256_add_ps(num6, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 1], xmm1);

    }
}

void matmul_vec_rb_4(float* xout, float* x, float* w, int n, int d) {
    __m256 num1, num2, num3, num4, num5, num6, num7, num8;
    __m128 xmm1;
    
    for (int i = 0; i < d; i += 4) {
        num5 = _mm256_setzero_ps();
        num6 = _mm256_setzero_ps();
        num7 = _mm256_setzero_ps();
        num8 = _mm256_setzero_ps();

        
        for (int j = 0; j < n; j += 8) {
            num3 = _mm256_loadu_ps(&x[j]);
            
            num2 = _mm256_loadu_ps(&w[i * n + j]);
            num5 = _mm256_fmadd_ps(num2, num3, num5);
            
            num2 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
            num6 = _mm256_fmadd_ps(num2, num3, num6);
   
            num2 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
            num7 = _mm256_fmadd_ps(num2, num3, num7);

            num2 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
            num8 = _mm256_fmadd_ps(num2, num3, num8);
          
        }
        
        num4 = _mm256_permute2f128_ps(num5, num5, 1);
        num1 = _mm256_add_ps(num5, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i], xmm1);
        
        num4 = _mm256_permute2f128_ps(num6, num6, 1);
        num1 = _mm256_add_ps(num6, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 1], xmm1);

        num4 = _mm256_permute2f128_ps(num7, num7, 1);
        num1 = _mm256_add_ps(num7, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 2], xmm1);

        num4 = _mm256_permute2f128_ps(num8, num8, 1);
        num1 = _mm256_add_ps(num8, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 3], xmm1);

    }
}

void matmul_vec_rb_8(float* xout, float* x, float* w, int n, int d) {
    __m256 num1, num2, num3, num4, num5, num6, num7, num8, num9, num10, num11, num12;
    __m128 xmm1;
    
    for (int i = 0; i < d; i += 8) {
        num5 = _mm256_setzero_ps();
        num6 = _mm256_setzero_ps();
        num7 = _mm256_setzero_ps();
        num8 = _mm256_setzero_ps();
        num9 = _mm256_setzero_ps();
        num10 = _mm256_setzero_ps();
        num11 = _mm256_setzero_ps();
        num12 = _mm256_setzero_ps();

        
        for (int j = 0; j < n; j += 8) {
            num3 = _mm256_loadu_ps(&x[j]);
            
            num2 = _mm256_loadu_ps(&w[i * n + j]);
            num5 = _mm256_fmadd_ps(num2, num3, num5);
            
            num2 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
            num6 = _mm256_fmadd_ps(num2, num3, num6);
   
            num2 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
            num7 = _mm256_fmadd_ps(num2, num3, num7);

            num2 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
            num8 = _mm256_fmadd_ps(num2, num3, num8);

            num2 = _mm256_loadu_ps(&w[(i + 4) * n + j]);
            num5 = _mm256_fmadd_ps(num2, num3, num5);
            
            num2 = _mm256_loadu_ps(&w[(i + 5) * n + j]);
            num6 = _mm256_fmadd_ps(num2, num3, num6);
   
            num2 = _mm256_loadu_ps(&w[(i + 6) * n + j]);
            num7 = _mm256_fmadd_ps(num2, num3, num7);

            num2 = _mm256_loadu_ps(&w[(i + 7) * n + j]);
            num8 = _mm256_fmadd_ps(num2, num3, num8);
          
        }
        
        num4 = _mm256_permute2f128_ps(num5, num5, 1);
        num1 = _mm256_add_ps(num5, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i], xmm1);
        
        num4 = _mm256_permute2f128_ps(num6, num6, 1);
        num1 = _mm256_add_ps(num6, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 1], xmm1);

        num4 = _mm256_permute2f128_ps(num7, num7, 1);
        num1 = _mm256_add_ps(num7, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 2], xmm1);

        num4 = _mm256_permute2f128_ps(num8, num8, 1);
        num1 = _mm256_add_ps(num8, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 3], xmm1);

        num4 = _mm256_permute2f128_ps(num9, num9, 1);
        num1 = _mm256_add_ps(num9, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 4], xmm1);
        
        num4 = _mm256_permute2f128_ps(num10, num10, 1);
        num1 = _mm256_add_ps(num10, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 5], xmm1);

        num4 = _mm256_permute2f128_ps(num11, num11, 1);
        num1 = _mm256_add_ps(num11, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 6], xmm1);

        num4 = _mm256_permute2f128_ps(num12, num12, 1);
        num1 = _mm256_add_ps(num12, num4);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 7], xmm1);

    }
}


// Loop Tiling
void matmul_loop_tiling(float* xout, float* x, float* w, int n, int d, int TILE){
    float val_x;

    for (int ii = 0; ii < d; ii += TILE){
        for (int i = ii; i < ii + TILE; i += 4) {
            float val1 = 0.0f;
            float val2 = 0.0f;
            float val3 = 0.0f;
            float val4 = 0.0f;
            
            for (int jj = 0; jj < n; jj += TILE){
                for (int j = jj; j < jj + TILE; j++) {
                    val_x = x[j];
        
                    val1 += w[i * n + j] * val_x;
                    val2 += w[(i + 1) * n + j] * val_x;
        
                    val3 += w[(i + 2) * n + j] * val_x;
                    val4 += w[(i + 3) * n + j] * val_x;
                }
            }
            
            xout[i] = val1;
            xout[i + 1] = val2;
            xout[i + 2] = val3;
            xout[i + 3] = val4;        
        }
    }
}

void matmul_vec_loop_tiling(float* xout, float* x, float* w, int n, int d, int TILE){
    __m256 num1, num2, num3, num4, num5, num6, num7, num8;
    __m128 xmm1;

    for (int ii = 0; ii < d; ii += TILE){
        for (int i = ii; i < ii + TILE; i += 4) {

            // Register blocking with size 4
            // Setting sum to 0
            num5 = _mm256_setzero_ps();
            num6 = _mm256_setzero_ps();
            num7 = _mm256_setzero_ps();
            num8 = _mm256_setzero_ps();

            for (int jj = 0; jj < n; jj += TILE){
                for (int j = jj; j < jj + TILE; j += 8) {

                    // loading 8 values of x into num3
                    num3 = _mm256_loadu_ps(&x[j]);
                    
                    // loading values of w and applying a fused multiply and add
                    num2 = _mm256_loadu_ps(&w[i * n + j]);
                    num5 = _mm256_fmadd_ps(num2, num3, num5);
                    
                    num2 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
                    num6 = _mm256_fmadd_ps(num2, num3, num6);
            
                    num2 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
                    num7 = _mm256_fmadd_ps(num2, num3, num7);

                    num2 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
                    num8 = _mm256_fmadd_ps(num2, num3, num8);
                    
                }
            }

            // accumulating the values and storing into xout
            num4 = _mm256_permute2f128_ps(num5, num5, 1);
            num1 = _mm256_add_ps(num5, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i], xmm1);
            
            num4 = _mm256_permute2f128_ps(num6, num6, 1);
            num1 = _mm256_add_ps(num6, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 1], xmm1);

            num4 = _mm256_permute2f128_ps(num7, num7, 1);
            num1 = _mm256_add_ps(num7, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 2], xmm1);

            num4 = _mm256_permute2f128_ps(num8, num8, 1);
            num1 = _mm256_add_ps(num8, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 3], xmm1);

        }
    }
}

// OMP


void matmul_omp(float* xout, float* x, float* w, int n, int d, int TILE){
    __m256 num1, num2, num3, num4, num5, num6, num7, num8;
    __m128 xmm1;

    int ii, i, jj, j;
    #pragma omp parallel
    {
    #pragma omp for private(i, jj, j, num1, num2, num3, num4, num5, num6, num7, num8, xmm1) schedule(static)
    for (ii = 0; ii < d; ii += TILE){
        for (i = ii; i < ii + TILE; i += 4) {

            // Register blocking with size 4
            // Setting sum to 0
            num5 = _mm256_setzero_ps();
            num6 = _mm256_setzero_ps();
            num7 = _mm256_setzero_ps();
            num8 = _mm256_setzero_ps();

            for (jj = 0; jj < n; jj += TILE){
                #pragma omp reduction(+:num5, num6, num7, num8)
                for (j = jj; j < jj + TILE; j += 8) {

                    // loading 8 values of x into num3
                    num3 = _mm256_loadu_ps(&x[j]);
                    
                    // loading values of w and applying a fused multiply and add
                    num2 = _mm256_loadu_ps(&w[i * n + j]);
                    num5 = _mm256_fmadd_ps(num2, num3, num5);
                    
                    num2 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
                    num6 = _mm256_fmadd_ps(num2, num3, num6);
            
                    num2 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
                    num7 = _mm256_fmadd_ps(num2, num3, num7);

                    num2 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
                    num8 = _mm256_fmadd_ps(num2, num3, num8);
                    
                }
            }

            // accumulating the values and storing into xout
            num4 = _mm256_permute2f128_ps(num5, num5, 1);
            num1 = _mm256_add_ps(num5, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i], xmm1);
            
            num4 = _mm256_permute2f128_ps(num6, num6, 1);
            num1 = _mm256_add_ps(num6, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 1], xmm1);

            num4 = _mm256_permute2f128_ps(num7, num7, 1);
            num1 = _mm256_add_ps(num7, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 2], xmm1);

            num4 = _mm256_permute2f128_ps(num8, num8, 1);
            num1 = _mm256_add_ps(num8, num4);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 3], xmm1);

        }
    }
    }
}

// Initialising an array to test the matmuls()

void initialise(){
    int i, j;
    for(i = 0; i < N; i++){
        xout[i] = (float) (i%7+0.01);
        x[i] = (float) (i%7+0.01);

        for(j = 0; j < N; j++){
            w[i * j] = (float) (i%7+0.01);

        }
    }  
}

int main() {
    double start, end;
    long long flops;

    initialise();

    omp_set_num_threads(8);
    start = omp_get_wtime();
    for (int i = 0; i < 16; i++){
        //Baseline
        //matmul(xout, x, w, N, N);

        //Vectorised
        //matmul_vectorised(xout, x, w, N, N);

        //Register Blocking
        //matmul_rb_2(xout, x, w, N, N);
        //matmul_rb_4(xout, x, w, N, N);
        //matmulrb_8(xout, x, w, N, N);

        //Vectorised Register Blocking
        //matmul_vec_rb_2(xout, x, w, N, N);
        //matmul_vec_rb_4(xout, x, w, N, N);
        //matmul_vec_rb_8(xout, x, w, N, N);

        //Loop Tiling
        //matmul_loop_tiling(xout, x, w, N, N, 16);

        //Vectorised Loop Tiling
        //matmul_vec_loop_tiling(xout, x, w, N, N, 8);

        //OMP Vectorised Loop Tiling
        //matmul_omp(xout, x, w, N, N, 64);
    }
    end = omp_get_wtime();
    flops = (2 * N * N);
    printf("Average Run Time: %f seconds\nGFLOPS calculated: %f", ((end-start)/16), ((16 * flops)/(end-start))/BILLION);

    return 0;
}