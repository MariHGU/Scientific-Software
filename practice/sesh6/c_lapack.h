#ifndef C_LAPACK_H
#define C_LAPACK_H

#ifdef __cplusplus
extern "C"{
#endif

void daxpy_(const int* n, const double* alpha, const double* x, const int* stridex, double* y, const int* stridey);


#ifdef __cplusplus
}
#endif

#endif