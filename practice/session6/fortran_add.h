#ifndef FORTRAN_ADD_H
#define FORTRAN_ADD_H

// If this is being included from a C++ file,
// we need to tell the compiler.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Calculates y = y + x
 *
 * @param n Length of the vectors
 * @param x Pointer to the first element of the vector x
 * @param y Pointer to the first element of the vector y
 *
 * @note The vectors x and y must have the same length
 *       and must be stored contiguously in memory
 */
void fortran_add_(const int* n, const double* x, double* y);

// Closing bracket for extern "C" block
#ifdef __cplusplus
}
#endif

#endif