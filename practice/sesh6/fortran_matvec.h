#ifndef FORTRAN_MATVEC_H
#define FORTRAN_MATVEC_H

#ifdef __cplusplus
extern "C"{
#endif


void fortran_matvec_(const int* m, const int* n, const double* A, const int* lda, const double* x, double* y);

#ifdef __cplusplus
}
#endif

#endif