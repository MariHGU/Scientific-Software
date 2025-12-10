#ifndef C_LAPACK_H
#define C_LAPACK_H

#ifdef __cplusplus
extern "C"{
#endif

// BLAS
void daxpy_(const int* n,
            const double* alpha,
            const double* x, const int* incx,
            double* y, const int* incy);

// LAPACK
void dgeqrf_(const int* m, const int* n,
             double* A, const int* lda,
             double* tau,
             double* work, const int* lwork,
             int* info);

void dormqr_(const char* side, const char* trans,
             const int* m, const int* n, const int* k,
             const double* A, const int* lda,
             const double* tau,
             double* C, const int* ldc,
             double* work, const int* lwork,
             int* info);

void dtrsm_(const char* side, const char* uplo,
            const char* transa, const char* diag,
            const int* m, const int* n,
            const double* alpha,
            const double* A, const int* lda,
            double* B, const int* ldb);

void dgemv_(const char* trans,
            const int* m, const int* n,
            const double* alpha,
            const double* A, const int* lda,
            const double* x, const int* incx,
            const double* beta,
            double* y, const int* incy);

#ifdef __cplusplus
}
#endif

#endif