#include "immintrin.h"
#include "c_add.h"

#ifdef __AVX512F__

// c_add using AVX512
void c_add(int n, const double* x, double* y)
{
    int i;
    __m512d x_vec, y_vec;
    for (i = 0; i < n; i += 8) {
        x_vec = _mm512_loadu_pd(x + i);
        y_vec = _mm512_loadu_pd(y + i);
        y_vec = _mm512_add_pd(y_vec, x_vec);
        _mm512_storeu_pd(y + i, y_vec);
    }
    for (; i < n; i++) {
        y[i] += x[i];
    }
}

#else

    #ifdef __AVX2__

// c_add using AVX2
void c_add(int n, const double* x, double* y)
{
    int i;
    __m256d x_vec, y_vec;
    for (i = 0; i < n; i += 4) {
        x_vec = _mm256_loadu_pd(x + i);
        y_vec = _mm256_loadu_pd(y + i);
        y_vec = _mm256_add_pd(y_vec, x_vec);
        _mm256_storeu_pd(y + i, y_vec);
    }
    for (; i < n; i++) {
        y[i] += x[i];
    }
}

    #else


// c_add using regular loops
void c_add(int n, const double* x, double* y)
{
    for (int i = 0; i < n; i++) {
        y[i] += x[i];
    }
}

    #endif

#endif
