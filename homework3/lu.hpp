#ifndef lu_hpp
#define lu_hpp

#include <cassert>

#include "matrix.hpp"
#include "vector.hpp"

#include "triangular_solve.hpp"
#include "matmul.hpp"

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

        // Straight forward fix: swap i and k (matricies are column-major)
        for (int i = j + 1; i < m; ++i){
            A(i, j) = A(i, j) / A(j, j);
        }
        for (int k = j + 1; k < n; ++k) {
            for (int i = j + 1; i < m; ++i) {
                A(i, k) -= A(i, j) * A(j, k);
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
    int min_val = std::min(A.num_rows(), A.num_columns());
    assert(ipiv.size() == min_val);

    int m = A.num_rows();
    int n = A.num_columns();

    int nb = 128; // block size
    for (int i =0; i < min_val; i += nb){
        //find width:
        int bi = std::min(nb, min_val - i);

        // .submatrix(start row, end row, start col, end col)
        auto A00_10 = A.submatrix(i, m, i, bi+i);
        auto ipiv_panel = ipiv.subvector(i, bi+i);
        lu_v2(A00_10, ipiv_panel); // -> Now A00 = L00 and U00 and A10 = L10 and U10

        
        // apply pivoting:
        //pivot index runs from 0 to bi-1
        for (int p = 0; p < bi; ++p)
        {
            int global_p = p + i; // global meaning complete A matrix
            int global_q = i + ipiv_panel[p]; // row that was swapped
            ipiv[global_p] = global_q; // store global pivot index
            
            if (global_p == global_q){
                continue; // no swap needed
            }
            
            // Apply swap globally, but only from columns to the right of A00
            for (int col = bi+i; col < n; ++col){
                std::swap(A(global_p, col), A(global_q, col));
            }

        }
        
        if(bi + i >= n){
            continue;
        }
        // retrieve pivoted submatricies
        auto A01 = A.submatrix(i, m, bi+i, n);
        auto A10 = A.submatrix(bi+i, m, i, bi+i); // L10 is in A10
        auto A11 = A.submatrix(bi+i, m, bi+i, n);
        
        // U01 = L00^-1 * A01
        auto L00 = A.submatrix(i, bi+i, i, bi+i); // L00 is the bi x bi square in A00
        auto U01 = A.submatrix(i, bi+i, i+bi, n); // Top bi rows of A01
        trsm_ll_v2(L00, U01); // Solution stored in U01
        
        // => A11 <- A11 - L10*U01
        tws::matmul_blocked_a(A10, U01, A11, T(-1), T(1));
    }
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
    int min_val = std::min(A.num_rows(), A.num_columns());
    assert(ipiv.size() == min_val);

    int m = A.num_rows();
    int n = A.num_cokumns();

    int nb = 128;
    // iterate through blocks
    for (int i = 0; i < min_val; i+=nb){

    }


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