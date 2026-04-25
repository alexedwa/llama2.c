#include <stdio.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <windows.h>

#define N 1024
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

// OMP
void omp_matmul_basic(float* xout, float* x, float* w, int n, int d, int TILE);
void omp_matmul_for(float* xout, float* x, float* w, int n, int d, int TILE);
void omp_matmul_full(float* xout, float* x, float* w, int n, int d, int TILE);


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

void matmul_rb_12(float* xout, float* x, float* w, int n, int d) {
    float val_x;
    for (int i = 0; i < d; i += 12) {
        float val1 = 0.0f;
        float val2 = 0.0f;
        float val3 = 0.0f;
        float val4 = 0.0f;
        float val5 = 0.0f;
        float val6 = 0.0f;
        float val7 = 0.0f;
        float val8 = 0.0f;
        float val9 = 0.0f;
        float val10 = 0.0f;
        float val11 = 0.0f;
        float val12 = 0.0f;
        
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

            val9 += w[(i + 8) * n + j] * val_x;
            val10 += w[(i + 9) * n + j] * val_x;

            val11 += w[(i + 10) * n + j] * val_x;
            val12 += w[(i + 11) * n + j] * val_x;
        }
        
        xout[i] = val1;
        xout[i + 1] = val2;
        xout[i + 2] = val3;
        xout[i + 3] = val4;

        xout[i + 4] = val5;
        xout[i + 5] = val6;
        xout[i + 6] = val7;
        xout[i + 7] = val8;

        xout[i + 8] = val5;
        xout[i + 9] = val6;
        xout[i + 10] = val7;
        xout[i + 11] = val8;
        
    }
}

void matmul_rb_16(float* xout, float* x, float* w, int n, int d) {
    float val_x;
    for (int i = 0; i < d; i += 16) {
        float val1 = 0.0f;
        float val2 = 0.0f;
        float val3 = 0.0f;
        float val4 = 0.0f;
        float val5 = 0.0f;
        float val6 = 0.0f;
        float val7 = 0.0f;
        float val8 = 0.0f;
        float val9 = 0.0f;
        float val10 = 0.0f;
        float val11 = 0.0f;
        float val12 = 0.0f;
        float val13 = 0.0f;
        float val14 = 0.0f;
        float val15 = 0.0f;
        float val16 = 0.0f;
        
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

            val9 += w[(i + 8) * n + j] * val_x;
            val10 += w[(i + 9) * n + j] * val_x;

            val11 += w[(i + 10) * n + j] * val_x;
            val12 += w[(i + 11) * n + j] * val_x;

            val13 += w[(i + 12) * n + j] * val_x;
            val14 += w[(i + 13) * n + j] * val_x;

            val15 += w[(i + 14) * n + j] * val_x;
            val16 += w[(i + 15) * n + j] * val_x;
        }
        
        xout[i] = val1;
        xout[i + 1] = val2;
        xout[i + 2] = val3;
        xout[i + 3] = val4;

        xout[i + 4] = val5;
        xout[i + 5] = val6;
        xout[i + 6] = val7;
        xout[i + 7] = val8;

        xout[i + 8] = val5;
        xout[i + 9] = val6;
        xout[i + 10] = val7;
        xout[i + 11] = val8;

        xout[i + 12] = val13;
        xout[i + 13] = val14;
        xout[i + 14] = val15;
        xout[i + 15] = val16;
        
    }
}

void matmul_rb_20(float* xout, float* x, float* w, int n, int d) {
    float val_x;
    for (int i = 0; i < d; i += 12) {
        float val1 = 0.0f;
        float val2 = 0.0f;
        float val3 = 0.0f;
        float val4 = 0.0f;
        float val5 = 0.0f;
        float val6 = 0.0f;
        float val7 = 0.0f;
        float val8 = 0.0f;
        float val9 = 0.0f;
        float val10 = 0.0f;
        float val11 = 0.0f;
        float val12 = 0.0f;
        float val13 = 0.0f;
        float val14 = 0.0f;
        float val15 = 0.0f;
        float val16 = 0.0f;
        float val17 = 0.0f;
        float val18 = 0.0f;
        float val19 = 0.0f;
        float val20 = 0.0f;
        
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

            val9 += w[(i + 8) * n + j] * val_x;
            val10 += w[(i + 9) * n + j] * val_x;

            val11 += w[(i + 10) * n + j] * val_x;
            val12 += w[(i + 11) * n + j] * val_x;

            val13 += w[(i + 12) * n + j] * val_x;
            val14 += w[(i + 13) * n + j] * val_x;

            val15 += w[(i + 14) * n + j] * val_x;
            val16 += w[(i + 15) * n + j] * val_x;

            val17 += w[(i + 16) * n + j] * val_x;
            val18 += w[(i + 17) * n + j] * val_x;

            val19 += w[(i + 18) * n + j] * val_x;
            val20 += w[(i + 19) * n + j] * val_x;
        }
        
        xout[i] = val1;
        xout[i + 1] = val2;
        xout[i + 2] = val3;
        xout[i + 3] = val4;

        xout[i + 4] = val5;
        xout[i + 5] = val6;
        xout[i + 6] = val7;
        xout[i + 7] = val8;

        xout[i + 8] = val5;
        xout[i + 9] = val6;
        xout[i + 10] = val7;
        xout[i + 11] = val8;

        xout[i + 12] = val13;
        xout[i + 13] = val14;
        xout[i + 14] = val15;
        xout[i + 15] = val16;

        xout[i + 16] = val17;
        xout[i + 17] = val18;
        xout[i + 18] = val19;
        xout[i + 19] = val20;
        
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
    // 13 + 2 registers needed

    __m256 val1, val2, val3, val4, val5, val6, val7, val8;
    __m256 num1, num2, xnum, wnum;
    __m128 xmm1;
    
    for (int i = 0; i < d; i += 8) {
        val1 = _mm256_setzero_ps();
        val2 = _mm256_setzero_ps();
        val3 = _mm256_setzero_ps();
        val4 = _mm256_setzero_ps();
        val5 = _mm256_setzero_ps();
        val6 = _mm256_setzero_ps();
        val7 = _mm256_setzero_ps();
        val8 = _mm256_setzero_ps();

        
        for (int j = 0; j < n; j += 8) {
            xnum = _mm256_loadu_ps(&x[j]);
            
            wnum = _mm256_loadu_ps(&w[i * n + j]);
            val1 = _mm256_fmadd_ps(wnum, xnum, val1);
            
            wnum = _mm256_loadu_ps(&w[(i + 1) * n + j]);
            val2 = _mm256_fmadd_ps(wnum, xnum, val2);

            wnum = _mm256_loadu_ps(&w[(i + 2) * n + j]);
            val3 = _mm256_fmadd_ps(wnum, xnum, val3);

            wnum = _mm256_loadu_ps(&w[(i + 3) * n + j]);
            val4 = _mm256_fmadd_ps(wnum, xnum, val4);

            wnum = _mm256_loadu_ps(&w[(i + 4) * n + j]);
            val5 = _mm256_fmadd_ps(wnum, xnum, val5);
            
            wnum = _mm256_loadu_ps(&w[(i + 5) * n + j]);
            val6 = _mm256_fmadd_ps(wnum, xnum, val6);

            wnum = _mm256_loadu_ps(&w[(i + 6) * n + j]);
            val7 = _mm256_fmadd_ps(wnum, xnum, val7);

            wnum = _mm256_loadu_ps(&w[(i + 7) * n + j]);
            val8 = _mm256_fmadd_ps(wnum, xnum, val8);
          
        }
        
        num2 = _mm256_permute2f128_ps(val1, val1, 1);
        num1 = _mm256_add_ps(val1, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i], xmm1);

        num2 = _mm256_permute2f128_ps(val2, val2, 1);
        num1 = _mm256_add_ps(val2, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 1], xmm1);

        num2 = _mm256_permute2f128_ps(val3, val3, 1);
        num1 = _mm256_add_ps(val3, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 2], xmm1);

        num2 = _mm256_permute2f128_ps(val4, val4, 1);
        num1 = _mm256_add_ps(val4, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 3], xmm1);

        num2 = _mm256_permute2f128_ps(val5, val5, 1);
        num1 = _mm256_add_ps(val5, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 4], xmm1);

        num2 = _mm256_permute2f128_ps(val6, val6, 1);
        num1 = _mm256_add_ps(val6, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 5], xmm1);

        num2 = _mm256_permute2f128_ps(val7, val7, 1);
        num1 = _mm256_add_ps(val7, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 6], xmm1);

        num2 = _mm256_permute2f128_ps(val8, val8, 1);
        num1 = _mm256_add_ps(val8, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 7], xmm1);

    }
}

void matmul_vec_rb_12(float* xout, float* x, float* w, int n, int d) {
    // 17 + 2 registers needed

    __m256 val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12;
    __m256 xnum, wnum, res1, res2;
    __m128 xmm1;
    int i, j;
    
    for (i = 0; i < d; i += 12) {
        val1 = _mm256_setzero_ps();
        val2 = _mm256_setzero_ps();
        val3 = _mm256_setzero_ps();
        val4 = _mm256_setzero_ps();

        val5 = _mm256_setzero_ps();
        val6 = _mm256_setzero_ps();
        val7 = _mm256_setzero_ps();
        val8 = _mm256_setzero_ps();

        val9 = _mm256_setzero_ps();
        val10 = _mm256_setzero_ps();
        val11 = _mm256_setzero_ps();
        val12 = _mm256_setzero_ps();

        
        for (j = 0; j < n; j += 8) {
            xnum = _mm256_loadu_ps(&x[j]);
            
            wnum = _mm256_loadu_ps(&w[i * n + j]);
            val1 = _mm256_fmadd_ps(wnum, xnum, val1);
            
            wnum = _mm256_loadu_ps(&w[(i + 1) * n + j]);
            val2 = _mm256_fmadd_ps(wnum, xnum, val2);

            wnum = _mm256_loadu_ps(&w[(i + 2) * n + j]);
            val3 = _mm256_fmadd_ps(wnum, xnum, val3);

            wnum = _mm256_loadu_ps(&w[(i + 3) * n + j]);
            val4 = _mm256_fmadd_ps(wnum, xnum, val4);

            wnum = _mm256_loadu_ps(&w[(i + 4) * n + j]);
            val5 = _mm256_fmadd_ps(wnum, xnum, val5);
            
            wnum = _mm256_loadu_ps(&w[(i + 5) * n + j]);
            val6 = _mm256_fmadd_ps(wnum, xnum, val6);

            wnum = _mm256_loadu_ps(&w[(i + 6) * n + j]);
            val7 = _mm256_fmadd_ps(wnum, xnum, val7);

            wnum = _mm256_loadu_ps(&w[(i + 7) * n + j]);
            val8 = _mm256_fmadd_ps(wnum, xnum, val8);

            wnum = _mm256_loadu_ps(&w[(i + 8) * n + j]);
            val9 = _mm256_fmadd_ps(wnum, xnum, val9);

            wnum = _mm256_loadu_ps(&w[(i + 9) * n + j]);
            val10 = _mm256_fmadd_ps(wnum, xnum, val10);

            wnum = _mm256_loadu_ps(&w[(i + 10) * n + j]);
            val11 = _mm256_fmadd_ps(wnum, xnum, val11);

            wnum = _mm256_loadu_ps(&w[(i + 11) * n + j]);
            val12 = _mm256_fmadd_ps(wnum, xnum, val12);
          
        }

        res1 = _mm256_permute2f128_ps(val1, val1, 1);
        res2 = _mm256_add_ps(val1, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i], xmm1);

        res1 = _mm256_permute2f128_ps(val2, val2, 1);
        res2 = _mm256_add_ps(val2, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 1], xmm1);

        res1 = _mm256_permute2f128_ps(val3, val3, 1);
        res2 = _mm256_add_ps(val3, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 2], xmm1);

        res1 = _mm256_permute2f128_ps(val4, val4, 1);
        res2 = _mm256_add_ps(val4, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 3], xmm1);

        res1 = _mm256_permute2f128_ps(val5, val5, 1);
        res2 = _mm256_add_ps(val5, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 4], xmm1);

        res1 = _mm256_permute2f128_ps(val6, val6, 1);
        res2 = _mm256_add_ps(val6, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 5], xmm1);

        res1 = _mm256_permute2f128_ps(val7, val7, 1);
        res2 = _mm256_add_ps(val7, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 6], xmm1);

        res1 = _mm256_permute2f128_ps(val8, val8, 1);
        res2 = _mm256_add_ps(val8, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 7], xmm1);

        res1 = _mm256_permute2f128_ps(val9, val9, 1);
        res2 = _mm256_add_ps(val9, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 8], xmm1);

        res1 = _mm256_permute2f128_ps(val10, val10, 1);
        res2 = _mm256_add_ps(val10, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 9], xmm1);

        res1 = _mm256_permute2f128_ps(val11, val11, 1);
        res2 = _mm256_add_ps(val11, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 10], xmm1);

        res1 = _mm256_permute2f128_ps(val12, val12, 1);
        res2 = _mm256_add_ps(val12, res1);
        res2 = _mm256_hadd_ps(res2, res2);
        res2 = _mm256_hadd_ps(res2, res2);
        xmm1 = _mm256_extractf128_ps(res2, 0);
        _mm_store_ss(&xout[i + 11], xmm1);

    }
    for (i = i; i < d; i++){
        float val = 0.0f;
        for(int j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmul_vec_rb_16(float* xout, float* x, float* w, int n, int d) {
    // 21 + 2 registers needed

    __m256 val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16;
    __m256 xnum1, wnum1, num1, num2;
    __m128 xmm1;
    int i, j;
    
    for (i = 0; i < d; i += 16) {
        val1 = _mm256_setzero_ps();
        val2 = _mm256_setzero_ps();
        val3 = _mm256_setzero_ps();
        val4 = _mm256_setzero_ps();

        val5 = _mm256_setzero_ps();
        val6 = _mm256_setzero_ps();
        val7 = _mm256_setzero_ps();
        val8 = _mm256_setzero_ps();

        val9 = _mm256_setzero_ps();
        val10 = _mm256_setzero_ps();
        val11 = _mm256_setzero_ps();
        val12 = _mm256_setzero_ps();

        val13 = _mm256_setzero_ps();
        val14 = _mm256_setzero_ps();
        val15 = _mm256_setzero_ps();
        val16 = _mm256_setzero_ps();
        
        for (j = 0; j < n; j += 8) {
            xnum1 = _mm256_loadu_ps(&x[j]);
            
            wnum1 = _mm256_loadu_ps(&w[i * n + j]);
            val1 = _mm256_fmadd_ps(wnum1, xnum1, val1);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
            val2 = _mm256_fmadd_ps(wnum1, xnum1, val2);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
            val3 = _mm256_fmadd_ps(wnum1, xnum1, val3);

            wnum1 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
            val4 = _mm256_fmadd_ps(wnum1, xnum1, val4);



            wnum1 = _mm256_loadu_ps(&w[(i + 4) * n + j]);
            val5 = _mm256_fmadd_ps(wnum1, xnum1, val5);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 5) * n + j]);
            val6 = _mm256_fmadd_ps(wnum1, xnum1, val6);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 6) * n + j]);
            val7 = _mm256_fmadd_ps(wnum1, xnum1, val7);

            wnum1 = _mm256_loadu_ps(&w[(i + 7) * n + j]);
            val8 = _mm256_fmadd_ps(wnum1, xnum1, val8);



            wnum1 = _mm256_loadu_ps(&w[(i + 8) * n + j]);
            val9 = _mm256_fmadd_ps(wnum1, xnum1, val9);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 9) * n + j]);
            val10 = _mm256_fmadd_ps(wnum1, xnum1, val10);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 10) * n + j]);
            val11 = _mm256_fmadd_ps(wnum1, xnum1, val11);

            wnum1 = _mm256_loadu_ps(&w[(i + 11) * n + j]);
            val12 = _mm256_fmadd_ps(wnum1, xnum1, val12);



            wnum1 = _mm256_loadu_ps(&w[(i + 12) * n + j]);
            val13 = _mm256_fmadd_ps(wnum1, xnum1, val13);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 13) * n + j]);
            val14 = _mm256_fmadd_ps(wnum1, xnum1, val14);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 14) * n + j]);
            val15 = _mm256_fmadd_ps(wnum1, xnum1, val15);

            wnum1 = _mm256_loadu_ps(&w[(i + 15) * n + j]);
            val16 = _mm256_fmadd_ps(wnum1, xnum1, val16);
          
        }
        
        num2 = _mm256_permute2f128_ps(val1, val1, 1);
        num1 = _mm256_add_ps(val1, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i], xmm1);

        num2 = _mm256_permute2f128_ps(val2, val2, 1);
        num1 = _mm256_add_ps(val2, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 1], xmm1);

        num2 = _mm256_permute2f128_ps(val3, val3, 1);
        num1 = _mm256_add_ps(val3, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 2], xmm1);

        num2 = _mm256_permute2f128_ps(val4, val4, 1);
        num1 = _mm256_add_ps(val4, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 3], xmm1);



        num2 = _mm256_permute2f128_ps(val5, val5, 1);
        num1 = _mm256_add_ps(val5, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 4], xmm1);

        num2 = _mm256_permute2f128_ps(val6, val6, 1);
        num1 = _mm256_add_ps(val6, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 5], xmm1);

        num2 = _mm256_permute2f128_ps(val7, val7, 1);
        num1 = _mm256_add_ps(val7, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 6], xmm1);

        num2 = _mm256_permute2f128_ps(val8, val8, 1);
        num1 = _mm256_add_ps(val8, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 7], xmm1);



        num2 = _mm256_permute2f128_ps(val9, val9, 1);
        num1 = _mm256_add_ps(val9, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 8], xmm1);

        num2 = _mm256_permute2f128_ps(val10, val10, 1);
        num1 = _mm256_add_ps(val10, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 9], xmm1);

        num2 = _mm256_permute2f128_ps(val11, val11, 1);
        num1 = _mm256_add_ps(val11, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 10], xmm1);

        num2 = _mm256_permute2f128_ps(val12, val12, 1);
        num1 = _mm256_add_ps(val12, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 11], xmm1);
        


        num2 = _mm256_permute2f128_ps(val13, val13, 1);
        num1 = _mm256_add_ps(val13, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 12], xmm1);

        num2 = _mm256_permute2f128_ps(val14, val14, 1);
        num1 = _mm256_add_ps(val14, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 13], xmm1);

        num2 = _mm256_permute2f128_ps(val15, val15, 1);
        num1 = _mm256_add_ps(val15, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 14], xmm1);

        num2 = _mm256_permute2f128_ps(val16, val16, 1);
        num1 = _mm256_add_ps(val16, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 15], xmm1);
        

    }

    // CLEANUP LOOP
    for (i = i; i < d; i++){
        float val = 0.0f;
        for(j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void matmul_vec_rb_20(float* xout, float* x, float* w, int n, int d) {
    // 25 + 2 registers needed

    __m256 val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16, val17, val18, val19, val20;
    __m256 xnum1, wnum1, num1, num2;
    __m128 xmm1;
    int i, j;
    
    for (i = 0; i < d; i += 20) {
        val1 = _mm256_setzero_ps();
        val2 = _mm256_setzero_ps();
        val3 = _mm256_setzero_ps();
        val4 = _mm256_setzero_ps();

        val5 = _mm256_setzero_ps();
        val6 = _mm256_setzero_ps();
        val7 = _mm256_setzero_ps();
        val8 = _mm256_setzero_ps();

        val9 = _mm256_setzero_ps();
        val10 = _mm256_setzero_ps();
        val11 = _mm256_setzero_ps();
        val12 = _mm256_setzero_ps();

        val13 = _mm256_setzero_ps();
        val14 = _mm256_setzero_ps();
        val15 = _mm256_setzero_ps();
        val16 = _mm256_setzero_ps();

        val13 = _mm256_setzero_ps();
        val14 = _mm256_setzero_ps();
        val15 = _mm256_setzero_ps();
        val16 = _mm256_setzero_ps();

        val17 = _mm256_setzero_ps();
        val18 = _mm256_setzero_ps();
        val19 = _mm256_setzero_ps();
        val20 = _mm256_setzero_ps();
        
        for (j = 0; j < n; j += 8) {
            xnum1 = _mm256_loadu_ps(&x[j]);
            
            wnum1 = _mm256_loadu_ps(&w[i * n + j]);
            val1 = _mm256_fmadd_ps(wnum1, xnum1, val1);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
            val2 = _mm256_fmadd_ps(wnum1, xnum1, val2);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
            val3 = _mm256_fmadd_ps(wnum1, xnum1, val3);

            wnum1 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
            val4 = _mm256_fmadd_ps(wnum1, xnum1, val4);



            wnum1 = _mm256_loadu_ps(&w[(i + 4) * n + j]);
            val5 = _mm256_fmadd_ps(wnum1, xnum1, val5);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 5) * n + j]);
            val6 = _mm256_fmadd_ps(wnum1, xnum1, val6);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 6) * n + j]);
            val7 = _mm256_fmadd_ps(wnum1, xnum1, val7);

            wnum1 = _mm256_loadu_ps(&w[(i + 7) * n + j]);
            val8 = _mm256_fmadd_ps(wnum1, xnum1, val8);



            wnum1 = _mm256_loadu_ps(&w[(i + 8) * n + j]);
            val9 = _mm256_fmadd_ps(wnum1, xnum1, val9);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 9) * n + j]);
            val10 = _mm256_fmadd_ps(wnum1, xnum1, val10);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 10) * n + j]);
            val11 = _mm256_fmadd_ps(wnum1, xnum1, val11);

            wnum1 = _mm256_loadu_ps(&w[(i + 11) * n + j]);
            val12 = _mm256_fmadd_ps(wnum1, xnum1, val12);



            wnum1 = _mm256_loadu_ps(&w[(i + 12) * n + j]);
            val13 = _mm256_fmadd_ps(wnum1, xnum1, val13);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 13) * n + j]);
            val14 = _mm256_fmadd_ps(wnum1, xnum1, val14);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 14) * n + j]);
            val15 = _mm256_fmadd_ps(wnum1, xnum1, val15);

            wnum1 = _mm256_loadu_ps(&w[(i + 15) * n + j]);
            val16 = _mm256_fmadd_ps(wnum1, xnum1, val16);



            wnum1 = _mm256_loadu_ps(&w[(i + 16) * n + j]);
            val17 = _mm256_fmadd_ps(wnum1, xnum1, val17);
            
            wnum1 = _mm256_loadu_ps(&w[(i + 17) * n + j]);
            val18 = _mm256_fmadd_ps(wnum1, xnum1, val18);
   
            wnum1 = _mm256_loadu_ps(&w[(i + 18) * n + j]);
            val19 = _mm256_fmadd_ps(wnum1, xnum1, val19);

            wnum1 = _mm256_loadu_ps(&w[(i + 19) * n + j]);
            val20 = _mm256_fmadd_ps(wnum1, xnum1, val20);
          
        }
        
        num2 = _mm256_permute2f128_ps(val1, val1, 1);
        num1 = _mm256_add_ps(val1, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i], xmm1);

        num2 = _mm256_permute2f128_ps(val2, val2, 1);
        num1 = _mm256_add_ps(val2, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 1], xmm1);

        num2 = _mm256_permute2f128_ps(val3, val3, 1);
        num1 = _mm256_add_ps(val3, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 2], xmm1);

        num2 = _mm256_permute2f128_ps(val4, val4, 1);
        num1 = _mm256_add_ps(val4, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 3], xmm1);



        num2 = _mm256_permute2f128_ps(val5, val5, 1);
        num1 = _mm256_add_ps(val5, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 4], xmm1);

        num2 = _mm256_permute2f128_ps(val6, val6, 1);
        num1 = _mm256_add_ps(val6, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 5], xmm1);

        num2 = _mm256_permute2f128_ps(val7, val7, 1);
        num1 = _mm256_add_ps(val7, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 6], xmm1);

        num2 = _mm256_permute2f128_ps(val8, val8, 1);
        num1 = _mm256_add_ps(val8, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 7], xmm1);



        num2 = _mm256_permute2f128_ps(val9, val9, 1);
        num1 = _mm256_add_ps(val9, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 8], xmm1);

        num2 = _mm256_permute2f128_ps(val10, val10, 1);
        num1 = _mm256_add_ps(val10, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 9], xmm1);

        num2 = _mm256_permute2f128_ps(val11, val11, 1);
        num1 = _mm256_add_ps(val11, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 10], xmm1);

        num2 = _mm256_permute2f128_ps(val12, val12, 1);
        num1 = _mm256_add_ps(val12, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 11], xmm1);
        


        num2 = _mm256_permute2f128_ps(val13, val13, 1);
        num1 = _mm256_add_ps(val13, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 12], xmm1);

        num2 = _mm256_permute2f128_ps(val14, val14, 1);
        num1 = _mm256_add_ps(val14, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 13], xmm1);

        num2 = _mm256_permute2f128_ps(val15, val15, 1);
        num1 = _mm256_add_ps(val15, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 14], xmm1);

        num2 = _mm256_permute2f128_ps(val16, val16, 1);
        num1 = _mm256_add_ps(val16, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 15], xmm1);
        


        num2 = _mm256_permute2f128_ps(val17, val17, 1);
        num1 = _mm256_add_ps(val17, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 16], xmm1);

        num2 = _mm256_permute2f128_ps(val18, val18, 1);
        num1 = _mm256_add_ps(val18, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 17], xmm1);

        num2 = _mm256_permute2f128_ps(val19, val19, 1);
        num1 = _mm256_add_ps(val19, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 18], xmm1);

        num2 = _mm256_permute2f128_ps(val20, val20, 1);
        num1 = _mm256_add_ps(val20, num2);
        num1 = _mm256_hadd_ps(num1, num1);
        num1 = _mm256_hadd_ps(num1, num1);
        xmm1 = _mm256_extractf128_ps(num1, 0);
        _mm_store_ss(&xout[i + 19], xmm1);
    }

    // CLEANUP LOOP
    for (i = i; i < d; i++){
        float val = 0.0f;
        for(j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

// Loop Tiling
void matmul_loop_tiling(float* xout, float* x, float* w, int n, int d, int TILE) {
    int i, j;
    for (i = 0; i < d; i += TILE){
        float val = 0.0f;
        for (int ii = i; ii < d && ii < i + TILE; ii++){
            int j;
            for (j = 0; j < n; j += TILE){
                int jj;
                for (jj = j; jj < n && jj < j + TILE; jj++){
                    val += w[ii * n + jj] * x[jj];
                }
            }
        }
        xout[i] = val;
    }
    // CLEANUP LOOP
    for (i = i; i < d; i++){
        float val = 0.0f;
        for(j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


// Loop Tiling Vec
void matmul_vec_loop_tiling(float* xout, float* x, float* w, int n, int d, int TILE) {
    // 21 + 2 registers needed

    __m256 val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16;
    __m256 xnum1, wnum1, num1, num2;
    __m128 xmm1;
    int i, j, ii, jj;
    for (ii = 0; ii < d; ii += TILE){
        for (i = ii; i < d && i < ii + TILE; i += 16) {
            val1 = _mm256_setzero_ps();
            val2 = _mm256_setzero_ps();
            val3 = _mm256_setzero_ps();
            val4 = _mm256_setzero_ps();

            val5 = _mm256_setzero_ps();
            val6 = _mm256_setzero_ps();
            val7 = _mm256_setzero_ps();
            val8 = _mm256_setzero_ps();

            val9 = _mm256_setzero_ps();
            val10 = _mm256_setzero_ps();
            val11 = _mm256_setzero_ps();
            val12 = _mm256_setzero_ps();

            val13 = _mm256_setzero_ps();
            val14 = _mm256_setzero_ps();
            val15 = _mm256_setzero_ps();
            val16 = _mm256_setzero_ps();
            
            for (jj = 0; jj < n; jj += TILE){
                for (j = jj; j < n && j < jj + TILE; j += 8) {
                    xnum1 = _mm256_loadu_ps(&x[j]);
                    
                    wnum1 = _mm256_loadu_ps(&w[i * n + j]);
                    val1 = _mm256_fmadd_ps(wnum1, xnum1, val1);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
                    val2 = _mm256_fmadd_ps(wnum1, xnum1, val2);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
                    val3 = _mm256_fmadd_ps(wnum1, xnum1, val3);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
                    val4 = _mm256_fmadd_ps(wnum1, xnum1, val4);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 4) * n + j]);
                    val5 = _mm256_fmadd_ps(wnum1, xnum1, val5);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 5) * n + j]);
                    val6 = _mm256_fmadd_ps(wnum1, xnum1, val6);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 6) * n + j]);
                    val7 = _mm256_fmadd_ps(wnum1, xnum1, val7);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 7) * n + j]);
                    val8 = _mm256_fmadd_ps(wnum1, xnum1, val8);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 8) * n + j]);
                    val9 = _mm256_fmadd_ps(wnum1, xnum1, val9);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 9) * n + j]);
                    val10 = _mm256_fmadd_ps(wnum1, xnum1, val10);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 10) * n + j]);
                    val11 = _mm256_fmadd_ps(wnum1, xnum1, val11);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 11) * n + j]);
                    val12 = _mm256_fmadd_ps(wnum1, xnum1, val12);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 12) * n + j]);
                    val13 = _mm256_fmadd_ps(wnum1, xnum1, val13);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 13) * n + j]);
                    val14 = _mm256_fmadd_ps(wnum1, xnum1, val14);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 14) * n + j]);
                    val15 = _mm256_fmadd_ps(wnum1, xnum1, val15);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 15) * n + j]);
                    val16 = _mm256_fmadd_ps(wnum1, xnum1, val16);
                
                }
            }
            
            num2 = _mm256_permute2f128_ps(val1, val1, 1);
            num1 = _mm256_add_ps(val1, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i], xmm1);

            num2 = _mm256_permute2f128_ps(val2, val2, 1);
            num1 = _mm256_add_ps(val2, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 1], xmm1);

            num2 = _mm256_permute2f128_ps(val3, val3, 1);
            num1 = _mm256_add_ps(val3, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 2], xmm1);

            num2 = _mm256_permute2f128_ps(val4, val4, 1);
            num1 = _mm256_add_ps(val4, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 3], xmm1);



            num2 = _mm256_permute2f128_ps(val5, val5, 1);
            num1 = _mm256_add_ps(val5, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 4], xmm1);

            num2 = _mm256_permute2f128_ps(val6, val6, 1);
            num1 = _mm256_add_ps(val6, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 5], xmm1);

            num2 = _mm256_permute2f128_ps(val7, val7, 1);
            num1 = _mm256_add_ps(val7, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 6], xmm1);

            num2 = _mm256_permute2f128_ps(val8, val8, 1);
            num1 = _mm256_add_ps(val8, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 7], xmm1);



            num2 = _mm256_permute2f128_ps(val9, val9, 1);
            num1 = _mm256_add_ps(val9, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 8], xmm1);

            num2 = _mm256_permute2f128_ps(val10, val10, 1);
            num1 = _mm256_add_ps(val10, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 9], xmm1);

            num2 = _mm256_permute2f128_ps(val11, val11, 1);
            num1 = _mm256_add_ps(val11, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 10], xmm1);

            num2 = _mm256_permute2f128_ps(val12, val12, 1);
            num1 = _mm256_add_ps(val12, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 11], xmm1);
            


            num2 = _mm256_permute2f128_ps(val13, val13, 1);
            num1 = _mm256_add_ps(val13, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 12], xmm1);

            num2 = _mm256_permute2f128_ps(val14, val14, 1);
            num1 = _mm256_add_ps(val14, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 13], xmm1);

            num2 = _mm256_permute2f128_ps(val15, val15, 1);
            num1 = _mm256_add_ps(val15, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 14], xmm1);

            num2 = _mm256_permute2f128_ps(val16, val16, 1);
            num1 = _mm256_add_ps(val16, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 15], xmm1);
            

        }
    }

    // CLEANUP LOOP
    for (i = i; i < d; i++){
        float val = 0.0f;
        for(j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}


// OMP
void omp_matmul_basic(float* xout, float* x, float* w, int n, int d, int TILE) {
    #pragma omp parallel 
    {

    // 21 + 4 registers needed
    __m256 val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16;
    __m256 xnum1, wnum1, num1, num2;
    __m128 xmm1;
    int i, j, ii, jj;

    for (ii = 0; ii < d; ii += TILE){
        for (i = ii; i < d && i < ii + TILE; i += 16) {
            val1 = _mm256_setzero_ps();
            val2 = _mm256_setzero_ps();
            val3 = _mm256_setzero_ps();
            val4 = _mm256_setzero_ps();

            val5 = _mm256_setzero_ps();
            val6 = _mm256_setzero_ps();
            val7 = _mm256_setzero_ps();
            val8 = _mm256_setzero_ps();

            val9 = _mm256_setzero_ps();
            val10 = _mm256_setzero_ps();
            val11 = _mm256_setzero_ps();
            val12 = _mm256_setzero_ps();

            val13 = _mm256_setzero_ps();
            val14 = _mm256_setzero_ps();
            val15 = _mm256_setzero_ps();
            val16 = _mm256_setzero_ps();
            
            for (jj = 0; jj < n; jj += TILE){
                for (j = jj; j < n && j < jj + TILE; j += 8) {
                    xnum1 = _mm256_loadu_ps(&x[j]);
                    
                    wnum1 = _mm256_loadu_ps(&w[i * n + j]);
                    val1 = _mm256_fmadd_ps(wnum1, xnum1, val1);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
                    val2 = _mm256_fmadd_ps(wnum1, xnum1, val2);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
                    val3 = _mm256_fmadd_ps(wnum1, xnum1, val3);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
                    val4 = _mm256_fmadd_ps(wnum1, xnum1, val4);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 4) * n + j]);
                    val5 = _mm256_fmadd_ps(wnum1, xnum1, val5);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 5) * n + j]);
                    val6 = _mm256_fmadd_ps(wnum1, xnum1, val6);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 6) * n + j]);
                    val7 = _mm256_fmadd_ps(wnum1, xnum1, val7);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 7) * n + j]);
                    val8 = _mm256_fmadd_ps(wnum1, xnum1, val8);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 8) * n + j]);
                    val9 = _mm256_fmadd_ps(wnum1, xnum1, val9);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 9) * n + j]);
                    val10 = _mm256_fmadd_ps(wnum1, xnum1, val10);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 10) * n + j]);
                    val11 = _mm256_fmadd_ps(wnum1, xnum1, val11);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 11) * n + j]);
                    val12 = _mm256_fmadd_ps(wnum1, xnum1, val12);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 12) * n + j]);
                    val13 = _mm256_fmadd_ps(wnum1, xnum1, val13);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 13) * n + j]);
                    val14 = _mm256_fmadd_ps(wnum1, xnum1, val14);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 14) * n + j]);
                    val15 = _mm256_fmadd_ps(wnum1, xnum1, val15);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 15) * n + j]);
                    val16 = _mm256_fmadd_ps(wnum1, xnum1, val16);
                
                }
            }
            
            num2 = _mm256_permute2f128_ps(val1, val1, 1);
            num1 = _mm256_add_ps(val1, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i], xmm1);

            num2 = _mm256_permute2f128_ps(val2, val2, 1);
            num1 = _mm256_add_ps(val2, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 1], xmm1);

            num2 = _mm256_permute2f128_ps(val3, val3, 1);
            num1 = _mm256_add_ps(val3, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 2], xmm1);

            num2 = _mm256_permute2f128_ps(val4, val4, 1);
            num1 = _mm256_add_ps(val4, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 3], xmm1);



            num2 = _mm256_permute2f128_ps(val5, val5, 1);
            num1 = _mm256_add_ps(val5, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 4], xmm1);

            num2 = _mm256_permute2f128_ps(val6, val6, 1);
            num1 = _mm256_add_ps(val6, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 5], xmm1);

            num2 = _mm256_permute2f128_ps(val7, val7, 1);
            num1 = _mm256_add_ps(val7, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 6], xmm1);

            num2 = _mm256_permute2f128_ps(val8, val8, 1);
            num1 = _mm256_add_ps(val8, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 7], xmm1);



            num2 = _mm256_permute2f128_ps(val9, val9, 1);
            num1 = _mm256_add_ps(val9, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 8], xmm1);

            num2 = _mm256_permute2f128_ps(val10, val10, 1);
            num1 = _mm256_add_ps(val10, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 9], xmm1);

            num2 = _mm256_permute2f128_ps(val11, val11, 1);
            num1 = _mm256_add_ps(val11, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 10], xmm1);

            num2 = _mm256_permute2f128_ps(val12, val12, 1);
            num1 = _mm256_add_ps(val12, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 11], xmm1);
            


            num2 = _mm256_permute2f128_ps(val13, val13, 1);
            num1 = _mm256_add_ps(val13, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 12], xmm1);

            num2 = _mm256_permute2f128_ps(val14, val14, 1);
            num1 = _mm256_add_ps(val14, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 13], xmm1);

            num2 = _mm256_permute2f128_ps(val15, val15, 1);
            num1 = _mm256_add_ps(val15, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 14], xmm1);

            num2 = _mm256_permute2f128_ps(val16, val16, 1);
            num1 = _mm256_add_ps(val16, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 15], xmm1);
            

        }
    }
    
    for (i = (d / 8) * 8; i < d; i++){
        float val = 0.0f;
        for(j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
    }
}

void omp_matmul_for(float* xout, float* x, float* w, int n, int d, int TILE) {
    #pragma omp parallel 
    {

    // 21 + 4 registers needed
    __m256 val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16;
    __m256 xnum1, wnum1, num1, num2;
    __m128 xmm1;
    int i, j, ii, jj;

    #pragma omp for 
    for (ii = 0; ii < d; ii += TILE){
        for (i = ii; i < d && i < ii + TILE; i += 16) {
            val1 = _mm256_setzero_ps();
            val2 = _mm256_setzero_ps();
            val3 = _mm256_setzero_ps();
            val4 = _mm256_setzero_ps();

            val5 = _mm256_setzero_ps();
            val6 = _mm256_setzero_ps();
            val7 = _mm256_setzero_ps();
            val8 = _mm256_setzero_ps();

            val9 = _mm256_setzero_ps();
            val10 = _mm256_setzero_ps();
            val11 = _mm256_setzero_ps();
            val12 = _mm256_setzero_ps();

            val13 = _mm256_setzero_ps();
            val14 = _mm256_setzero_ps();
            val15 = _mm256_setzero_ps();
            val16 = _mm256_setzero_ps();
            
            for (jj = 0; jj < n; jj += TILE){
                for (j = jj; j < n && j < jj + TILE; j += 8) {
                    xnum1 = _mm256_loadu_ps(&x[j]);
                    
                    wnum1 = _mm256_loadu_ps(&w[i * n + j]);
                    val1 = _mm256_fmadd_ps(wnum1, xnum1, val1);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
                    val2 = _mm256_fmadd_ps(wnum1, xnum1, val2);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
                    val3 = _mm256_fmadd_ps(wnum1, xnum1, val3);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
                    val4 = _mm256_fmadd_ps(wnum1, xnum1, val4);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 4) * n + j]);
                    val5 = _mm256_fmadd_ps(wnum1, xnum1, val5);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 5) * n + j]);
                    val6 = _mm256_fmadd_ps(wnum1, xnum1, val6);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 6) * n + j]);
                    val7 = _mm256_fmadd_ps(wnum1, xnum1, val7);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 7) * n + j]);
                    val8 = _mm256_fmadd_ps(wnum1, xnum1, val8);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 8) * n + j]);
                    val9 = _mm256_fmadd_ps(wnum1, xnum1, val9);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 9) * n + j]);
                    val10 = _mm256_fmadd_ps(wnum1, xnum1, val10);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 10) * n + j]);
                    val11 = _mm256_fmadd_ps(wnum1, xnum1, val11);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 11) * n + j]);
                    val12 = _mm256_fmadd_ps(wnum1, xnum1, val12);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 12) * n + j]);
                    val13 = _mm256_fmadd_ps(wnum1, xnum1, val13);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 13) * n + j]);
                    val14 = _mm256_fmadd_ps(wnum1, xnum1, val14);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 14) * n + j]);
                    val15 = _mm256_fmadd_ps(wnum1, xnum1, val15);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 15) * n + j]);
                    val16 = _mm256_fmadd_ps(wnum1, xnum1, val16);
                
                }
            }
            
            num2 = _mm256_permute2f128_ps(val1, val1, 1);
            num1 = _mm256_add_ps(val1, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i], xmm1);

            num2 = _mm256_permute2f128_ps(val2, val2, 1);
            num1 = _mm256_add_ps(val2, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 1], xmm1);

            num2 = _mm256_permute2f128_ps(val3, val3, 1);
            num1 = _mm256_add_ps(val3, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 2], xmm1);

            num2 = _mm256_permute2f128_ps(val4, val4, 1);
            num1 = _mm256_add_ps(val4, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 3], xmm1);



            num2 = _mm256_permute2f128_ps(val5, val5, 1);
            num1 = _mm256_add_ps(val5, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 4], xmm1);

            num2 = _mm256_permute2f128_ps(val6, val6, 1);
            num1 = _mm256_add_ps(val6, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 5], xmm1);

            num2 = _mm256_permute2f128_ps(val7, val7, 1);
            num1 = _mm256_add_ps(val7, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 6], xmm1);

            num2 = _mm256_permute2f128_ps(val8, val8, 1);
            num1 = _mm256_add_ps(val8, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 7], xmm1);



            num2 = _mm256_permute2f128_ps(val9, val9, 1);
            num1 = _mm256_add_ps(val9, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 8], xmm1);

            num2 = _mm256_permute2f128_ps(val10, val10, 1);
            num1 = _mm256_add_ps(val10, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 9], xmm1);

            num2 = _mm256_permute2f128_ps(val11, val11, 1);
            num1 = _mm256_add_ps(val11, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 10], xmm1);

            num2 = _mm256_permute2f128_ps(val12, val12, 1);
            num1 = _mm256_add_ps(val12, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 11], xmm1);
            


            num2 = _mm256_permute2f128_ps(val13, val13, 1);
            num1 = _mm256_add_ps(val13, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 12], xmm1);

            num2 = _mm256_permute2f128_ps(val14, val14, 1);
            num1 = _mm256_add_ps(val14, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 13], xmm1);

            num2 = _mm256_permute2f128_ps(val15, val15, 1);
            num1 = _mm256_add_ps(val15, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 14], xmm1);

            num2 = _mm256_permute2f128_ps(val16, val16, 1);
            num1 = _mm256_add_ps(val16, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 15], xmm1);
            

        }
    }
    

    // CLEANUP LOOP
    #pragma omp for
    for (i = (d / 8) * 8; i < d; i++){
        float val = 0.0f;
        for(j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
    }
}

void omp_matmul_full(float* xout, float* x, float* w, int n, int d, int TILE) {
    #pragma omp parallel 
    {

    // 21 + 4 registers needed
    __m256 val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15, val16;
    __m256 xnum1, wnum1, num1, num2;
    __m128 xmm1;
    int i, j, ii, jj;

    #pragma omp for schedule(static) nowait
    for (ii = 0; ii < d; ii += TILE){
        for (i = ii; i < d && i < ii + TILE; i += 16) {
            val1 = _mm256_setzero_ps();
            val2 = _mm256_setzero_ps();
            val3 = _mm256_setzero_ps();
            val4 = _mm256_setzero_ps();

            val5 = _mm256_setzero_ps();
            val6 = _mm256_setzero_ps();
            val7 = _mm256_setzero_ps();
            val8 = _mm256_setzero_ps();

            val9 = _mm256_setzero_ps();
            val10 = _mm256_setzero_ps();
            val11 = _mm256_setzero_ps();
            val12 = _mm256_setzero_ps();

            val13 = _mm256_setzero_ps();
            val14 = _mm256_setzero_ps();
            val15 = _mm256_setzero_ps();
            val16 = _mm256_setzero_ps();
            
            for (jj = 0; jj < n; jj += TILE){
                for (j = jj; j < n && j < jj + TILE; j += 8) {
                    xnum1 = _mm256_loadu_ps(&x[j]);
                    
                    wnum1 = _mm256_loadu_ps(&w[i * n + j]);
                    val1 = _mm256_fmadd_ps(wnum1, xnum1, val1);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 1) * n + j]);
                    val2 = _mm256_fmadd_ps(wnum1, xnum1, val2);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 2) * n + j]);
                    val3 = _mm256_fmadd_ps(wnum1, xnum1, val3);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 3) * n + j]);
                    val4 = _mm256_fmadd_ps(wnum1, xnum1, val4);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 4) * n + j]);
                    val5 = _mm256_fmadd_ps(wnum1, xnum1, val5);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 5) * n + j]);
                    val6 = _mm256_fmadd_ps(wnum1, xnum1, val6);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 6) * n + j]);
                    val7 = _mm256_fmadd_ps(wnum1, xnum1, val7);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 7) * n + j]);
                    val8 = _mm256_fmadd_ps(wnum1, xnum1, val8);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 8) * n + j]);
                    val9 = _mm256_fmadd_ps(wnum1, xnum1, val9);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 9) * n + j]);
                    val10 = _mm256_fmadd_ps(wnum1, xnum1, val10);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 10) * n + j]);
                    val11 = _mm256_fmadd_ps(wnum1, xnum1, val11);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 11) * n + j]);
                    val12 = _mm256_fmadd_ps(wnum1, xnum1, val12);
    
    
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 12) * n + j]);
                    val13 = _mm256_fmadd_ps(wnum1, xnum1, val13);
                    
                    wnum1 = _mm256_loadu_ps(&w[(i + 13) * n + j]);
                    val14 = _mm256_fmadd_ps(wnum1, xnum1, val14);
        
                    wnum1 = _mm256_loadu_ps(&w[(i + 14) * n + j]);
                    val15 = _mm256_fmadd_ps(wnum1, xnum1, val15);
    
                    wnum1 = _mm256_loadu_ps(&w[(i + 15) * n + j]);
                    val16 = _mm256_fmadd_ps(wnum1, xnum1, val16);
                
                }
            }
            
            num2 = _mm256_permute2f128_ps(val1, val1, 1);
            num1 = _mm256_add_ps(val1, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i], xmm1);

            num2 = _mm256_permute2f128_ps(val2, val2, 1);
            num1 = _mm256_add_ps(val2, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 1], xmm1);

            num2 = _mm256_permute2f128_ps(val3, val3, 1);
            num1 = _mm256_add_ps(val3, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 2], xmm1);

            num2 = _mm256_permute2f128_ps(val4, val4, 1);
            num1 = _mm256_add_ps(val4, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 3], xmm1);



            num2 = _mm256_permute2f128_ps(val5, val5, 1);
            num1 = _mm256_add_ps(val5, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 4], xmm1);

            num2 = _mm256_permute2f128_ps(val6, val6, 1);
            num1 = _mm256_add_ps(val6, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 5], xmm1);

            num2 = _mm256_permute2f128_ps(val7, val7, 1);
            num1 = _mm256_add_ps(val7, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 6], xmm1);

            num2 = _mm256_permute2f128_ps(val8, val8, 1);
            num1 = _mm256_add_ps(val8, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 7], xmm1);



            num2 = _mm256_permute2f128_ps(val9, val9, 1);
            num1 = _mm256_add_ps(val9, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 8], xmm1);

            num2 = _mm256_permute2f128_ps(val10, val10, 1);
            num1 = _mm256_add_ps(val10, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 9], xmm1);

            num2 = _mm256_permute2f128_ps(val11, val11, 1);
            num1 = _mm256_add_ps(val11, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 10], xmm1);

            num2 = _mm256_permute2f128_ps(val12, val12, 1);
            num1 = _mm256_add_ps(val12, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 11], xmm1);
            


            num2 = _mm256_permute2f128_ps(val13, val13, 1);
            num1 = _mm256_add_ps(val13, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 12], xmm1);

            num2 = _mm256_permute2f128_ps(val14, val14, 1);
            num1 = _mm256_add_ps(val14, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 13], xmm1);

            num2 = _mm256_permute2f128_ps(val15, val15, 1);
            num1 = _mm256_add_ps(val15, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 14], xmm1);

            num2 = _mm256_permute2f128_ps(val16, val16, 1);
            num1 = _mm256_add_ps(val16, num2);
            num1 = _mm256_hadd_ps(num1, num1);
            num1 = _mm256_hadd_ps(num1, num1);
            xmm1 = _mm256_extractf128_ps(num1, 0);
            _mm_store_ss(&xout[i + 15], xmm1);
            

        }
    }
    

    // CLEANUP LOOP
    #pragma omp for schedule(static) nowait
    for (i = (d / 8) * 8; i < d; i++){
        float val = 0.0f;
        for(j = 0; j < n; j++){
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
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
    int reruns = 1000;

    // Performance cores (0, 2, 4, 6) 0x55
    // Hyperthreaded cores (1, 3, 5, 7) 0xAA
    // Efficient cores (8, 9, 10, 11) 0xF00
    
    /*
    DWORD_PTR mask = (0xF00);
    
    if(!SetThreadAffinityMask(GetCurrentThread(), mask)) {
        fprintf(stderr, "Error setting thread affinity mask: %lu\n", GetLastError());
        return 1;
    }
    */
    //initialise();
    
    omp_matmul_full(xout, x, w, N, N, 32);
    //omp_set_num_threads(4);
    for(int j = 0; j < 1; j++){
        initialise();
        start = omp_get_wtime();
        for (int i = 0; i < reruns; i++){
            //Baseline
            //matmul(xout, x, w, N, N);

            //Vectorised
            //matmul_vectorised(xout, x, w, N, N);

            //Register Blocking
            //matmul_rb_2(xout, x, w, N, N);
            //matmul_rb_4(xout, x, w, N, N);
            //matmul_rb_8(xout, x, w, N, N);
            //matmul_rb_12(xout, x, w, N, N);
            //matmul_rb_16(xout, x, w, N, N);
            //matmul_rb_20(xout, x, w, N, N);

            //Vec + RB
            //matmul_vec_rb_2(xout, x, w, N, N);
            //matmul_vec_rb_4(xout, x, w, N, N);
            //matmul_vec_rb_8(xout, x, w, N, N);
            //matmul_vec_rb_12(xout, x, w, N, N);
            //matmul_vec_rb_16(xout, x, w, N, N);
            //matmul_vec_rb_20(xout, x, w, N, N);

            //Loop tiling
            matmul_loop_tiling(xout, x, w, N, N, 64);

            //Vec + RB + LT
            //matmul_vec_loop_tiling(xout, x, w, N, N, 64);

            //OMP Vectorised Loop Tiling
            //omp_matmul_basic(xout, x, w, N, N, 32);
            //omp_matmul_for(xout, x, w, N, N, 32);
            //omp_matmul_full(xout, x, w, N, N, 32);
        }
        end = omp_get_wtime();
        flops = (2 * N * N);
        printf("\nTotal Run Time: %f seconds\n", (end-start));
        printf("Average Run Time: %f seconds\n", ((end-start) / reruns));
        printf("GFLOPS: %f\n\n", ((reruns * flops)/(end-start))/BILLION);
    }
    return 0;
}
