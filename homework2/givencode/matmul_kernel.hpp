// This code is an adaptation https://github.com/flame/how-to-optimize-gemm
// Author: Thijs Steel, 2024

//
// DO NOT MODIFY THIS FILE
//

#ifndef matmul_kernel_hpp
#define matmul_kernel_hpp

// import ssimd intrinsics
#include <immintrin.h>

namespace tws {

/**
 * Calculates the matrix-matrix product C = C + alpha*A*B.
 * (note the lack of a beta parameter)
 *
 * This function has hard-coded some of the sizes, enabling
 * high performance.
 *
 * @param k      int
 *               The number of columns of A and rows of B
 * @param alpha  double
 *               The scalar to multiply the product with
 * @param A      8-by-k matrix
 *               The left matrix, stored in column-major order
 *               with leading dimension lda.
 * @param lda    int
 *               The leading dimension of A
 * @param B      k-by-6 matrix
 *               The right matrix, stored in column-major order
 *               with leading dimension ldb.
 * @param ldb    int
 *               The leading dimension of B.
 * @param C      8-by-6 matrix
 *               The result matrix, stored in column-major order
 *               with leading dimension ldc.
 * @param ldc    int
 *               The leading dimension of C.
 */
inline void kernel_8x6(int k,
                       double alpha,
                       const double* A,
                       int lda,
                       const double* B,
                       int ldb,
                       double* C,
                       int ldc)
{
    __m256d c0123_0 = _mm256_setzero_pd();
    __m256d c4567_0 = _mm256_setzero_pd();

    __m256d c0123_1 = _mm256_setzero_pd();
    __m256d c4567_1 = _mm256_setzero_pd();

    __m256d c0123_2 = _mm256_setzero_pd();
    __m256d c4567_2 = _mm256_setzero_pd();

    __m256d c0123_3 = _mm256_setzero_pd();
    __m256d c4567_3 = _mm256_setzero_pd();

    __m256d c0123_4 = _mm256_setzero_pd();
    __m256d c4567_4 = _mm256_setzero_pd();

    __m256d c0123_5 = _mm256_setzero_pd();
    __m256d c4567_5 = _mm256_setzero_pd();

    for (int p = 0; p < k; p++) {
        __m256d a0123 = _mm256_loadu_pd(&A[p * lda]);
        __m256d a4567 = _mm256_loadu_pd(&A[4 + p * lda]);

        __m256d b;

        b = _mm256_set1_pd(B[p]);
        c0123_0 = _mm256_fmadd_pd(a0123, b, c0123_0);
        c4567_0 = _mm256_fmadd_pd(a4567, b, c4567_0);

        b = _mm256_set1_pd(B[p + ldb]);
        c0123_1 = _mm256_fmadd_pd(a0123, b, c0123_1);
        c4567_1 = _mm256_fmadd_pd(a4567, b, c4567_1);

        b = _mm256_set1_pd(B[p + 2 * ldb]);
        c0123_2 = _mm256_fmadd_pd(a0123, b, c0123_2);
        c4567_2 = _mm256_fmadd_pd(a4567, b, c4567_2);

        b = _mm256_set1_pd(B[p + 3 * ldb]);
        c0123_3 = _mm256_fmadd_pd(a0123, b, c0123_3);
        c4567_3 = _mm256_fmadd_pd(a4567, b, c4567_3);

        b = _mm256_set1_pd(B[p + 4 * ldb]);
        c0123_4 = _mm256_fmadd_pd(a0123, b, c0123_4);
        c4567_4 = _mm256_fmadd_pd(a4567, b, c4567_4);

        b = _mm256_set1_pd(B[p + 5 * ldb]);
        c0123_5 = _mm256_fmadd_pd(a0123, b, c0123_5);
        c4567_5 = _mm256_fmadd_pd(a4567, b, c4567_5);
    }

    // Broadcast alpha so we can multiply it with the result
    __m256d alpha_v = _mm256_set1_pd(alpha);

    // Store the result back in C
    __m256d c_0123_temp;
    __m256d c_4567_temp;

    c_0123_temp = _mm256_loadu_pd(&C[0]);
    c_4567_temp = _mm256_loadu_pd(&C[4]);
    c_0123_temp = _mm256_fmadd_pd(alpha_v, c0123_0, c_0123_temp);
    c_4567_temp = _mm256_fmadd_pd(alpha_v, c4567_0, c_4567_temp);
    _mm256_storeu_pd(&C[0], c_0123_temp);
    _mm256_storeu_pd(&C[4], c_4567_temp);

    c_0123_temp = _mm256_loadu_pd(&C[ldc]);
    c_4567_temp = _mm256_loadu_pd(&C[4 + ldc]);
    c_0123_temp = _mm256_fmadd_pd(alpha_v, c0123_1, c_0123_temp);
    c_4567_temp = _mm256_fmadd_pd(alpha_v, c4567_1, c_4567_temp);
    _mm256_storeu_pd(&C[ldc], c_0123_temp);
    _mm256_storeu_pd(&C[4 + ldc], c_4567_temp);

    c_0123_temp = _mm256_loadu_pd(&C[2 * ldc]);
    c_4567_temp = _mm256_loadu_pd(&C[4 + 2 * ldc]);
    c_0123_temp = _mm256_fmadd_pd(alpha_v, c0123_2, c_0123_temp);
    c_4567_temp = _mm256_fmadd_pd(alpha_v, c4567_2, c_4567_temp);
    _mm256_storeu_pd(&C[2 * ldc], c_0123_temp);
    _mm256_storeu_pd(&C[4 + 2 * ldc], c_4567_temp);

    c_0123_temp = _mm256_loadu_pd(&C[3 * ldc]);
    c_4567_temp = _mm256_loadu_pd(&C[4 + 3 * ldc]);
    c_0123_temp = _mm256_fmadd_pd(alpha_v, c0123_3, c_0123_temp);
    c_4567_temp = _mm256_fmadd_pd(alpha_v, c4567_3, c_4567_temp);
    _mm256_storeu_pd(&C[3 * ldc], c_0123_temp);
    _mm256_storeu_pd(&C[4 + 3 * ldc], c_4567_temp);

    c_0123_temp = _mm256_loadu_pd(&C[4 * ldc]);
    c_4567_temp = _mm256_loadu_pd(&C[4 + 4 * ldc]);
    c_0123_temp = _mm256_fmadd_pd(alpha_v, c0123_4, c_0123_temp);
    c_4567_temp = _mm256_fmadd_pd(alpha_v, c4567_4, c_4567_temp);
    _mm256_storeu_pd(&C[4 * ldc], c_0123_temp);
    _mm256_storeu_pd(&C[4 + 4 * ldc], c_4567_temp);

    c_0123_temp = _mm256_loadu_pd(&C[5 * ldc]);
    c_4567_temp = _mm256_loadu_pd(&C[4 + 5 * ldc]);
    c_0123_temp = _mm256_fmadd_pd(alpha_v, c0123_5, c_0123_temp);
    c_4567_temp = _mm256_fmadd_pd(alpha_v, c4567_5, c_4567_temp);
    _mm256_storeu_pd(&C[5 * ldc], c_0123_temp);
    _mm256_storeu_pd(&C[4 + 5 * ldc], c_4567_temp);
}

}  // namespace tws

#endif