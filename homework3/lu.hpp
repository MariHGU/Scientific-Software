#ifndef lu_hpp
#define lu_hpp

#include <cassert>

#include "matrix.hpp"
#include "vector.hpp"

namespace tws {


/** 
 * Calculates the LU decomposition with partial pivoting of a nonsingular matrix A.
 *
 * The LU decomposition is stored in A, where the lower triangular part
 * of A contains the lower triangular matrix L and the upper triangular part of
 * A contains the upper triangular matrix U.
 * 
 * This is the initial version of the LU decomposition provided by the assistants.
 * NOTE: do not change this function.
 *
 * @param A [in,out] size m x n matrix
 *                   On input, the matrix A.
 *                   On output,
 *                   The strictly lower triangular part of A contains
 *                   the lower triangular matrix L. The diagonal of L is 1,
 *                   so we don't store it. The upper triangular part of A
 *                   contains the upper triangular matrix U.
 *                   Note:
 *                   If m > n, L is lower trapezoidal instead.
 *                   If m < n, U is upper trapezoidal instead.
 * 
 * @param ipiv [out] size min(m,n) vector of integers
 *                   On output, the vector ipiv contains the permutation of the
 *                   rows of A. The permutation is encoded as follows:
 *                   ipiv[i] = j means that the i-th row of A has been swapped
 *                   with the j-th row of A.
 *                   Example code to apply P to a matrix B :
 *                   for(int i = 0; i < m; ++i){ for(int j = 0; j < n; ++j){
 *                          std::swap(B(i,j), B(ipiv[i],j));
 *                       }
 *                   }
 *                   Example code to apply P**T to a matrix B :
 *                   for(int i = m-1; i >= 0; --i){
 *                      for(int j = 0; j < n; ++j){
 *                         std::swap(B(i,j), B(ipiv[i],j));
 *                      }
 *                   }
 */
template <typename T>
void lu_v1(matrixview<T> A, vectorview<int> ipiv)
{
    assert(ipiv.size() == std::min(A.num_rows(), A.num_columns()));

    int m = A.num_rows();
    int n = A.num_columns();
    for (int j = 0; j < std::min(m, n); ++j) {
        // Find the pivot element.
        // We take the largest element in the column.
        int pivot = j;
        for (int i = j + 1; i < m; ++i) {
            if (std::abs(A(i, j)) > std::abs(A(pivot, j))) {
                pivot = i;
            }
        }
        ipiv[j] = pivot;

        // Apply the row swap to A.
        if (j != pivot) {
            for (int k = 0; k < n; ++k) {
                std::swap(A(j, k), A(pivot, k));
            }
        }

        for (int i = j + 1; i < m; ++i) {
            A(i, j) = A(i, j) / A(j, j);
            for (int k = j + 1; k < n; ++k) {
                A(i, k) = A(i, k) - A(i, j) * A(j, k);
            }
        }
    }
}

/**
 * See the documentation of the lu function.
 *
 * Slightly optimized version of the LU decomposition.
 */
template <typename T>
void lu_v2(matrixview<T> A, vectorview<int> ipiv)
{
    assert(ipiv.size() == std::min(A.num_rows(), A.num_columns()));

    int m = A.num_rows();
    int n = A.num_columns();
    for (int j = 0; j < std::min(m, n); ++j) {
        // Find the pivot element.
        // We take the largest element in the column.
        int pivot = j;
        for (int i = j + 1; i < m; ++i) {
            if (std::abs(A(i, j)) > std::abs(A(pivot, j))) {
                pivot = i;
            }
        }
        ipiv[j] = pivot;

        // Apply the row swap to A.
        if (j != pivot) {
            for (int k = 0; k < n; ++k) {
                std::swap(A(j, k), A(pivot, k));
            }
        }

        for (int i = j + 1; i < m; ++i) {
            A(i, j) = A(i, j) / A(j, j);
            for (int k = j + 1; k < n; ++k) {
                A(i, k) = A(i, k) - A(i, j) * A(j, k);
            }
        }
    }
}

/**
 * See the documentation of the lu function.
 *
 * Blocked version of the LU decomposition.
 */
template <typename T>
void lu_v3(matrixview<T> A, vectorview<int> ipiv)
{
    // TODO: implement this function
}

/**
 * See the documentation of the lu function.
 *
 * Blocked version of the LU decomposition.
 * Uses efficient BLAS implementations where possible.
 */
template <typename T>
void lu_v4(matrixview<T> A, vectorview<int> ipiv)
{
    // TODO: implement this function
}

/**
 * See the documentation of the lu function.
 *
 * Wrapper around the LU factorization from LAPACK.
 */
template <typename T>
void lu_lapack(matrixview<T> A, vectorview<int> ipiv)
{
    // TODO: implement this function
}

}  // namespace tws

#endif