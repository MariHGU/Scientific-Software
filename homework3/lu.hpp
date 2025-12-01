#ifndef lu_hpp
#define lu_hpp

#include <cassert>
#include <cblas.h>
#include <type_traits>

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
            
            for (int col = 0; col < n; ++col){
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
    // same as v3 but use BLAS where possible
    static_assert(std::is_same_v<T,float> || std::is_same_v<T, double>, "lu_v4: type not supported for BLAS implementation");
    int min_val = std::min(A.num_rows(), A.num_columns());
    assert(ipiv.size() == min_val);

    int m = A.num_rows();
    int n = A.num_columns();

    int nb = 128;
    // iterate through blocks
    for (int i = 0; i < min_val; i+=nb){
        // find block size
        int bi = std::min(nb, min_val-i);

        auto A00_10 = A.submatrix(i, m, i, bi+i);
        auto ipiv_panel = ipiv.subvector(i, bi+i);
        lu_v2(A00_10, ipiv_panel); // LU factorization of left panel

        // apply pivoting:
        for (int p = 0; p < bi; ++p){
            int global_p = p + i;
            int global_q = i + ipiv_panel[p];

            ipiv[global_p] = global_q;

            if (global_p == global_q){
                continue;
            }

            for (int col =bi+i; col<n; ++col){
                std::swap(A(global_p, col), A(global_q, col));
            }
        }

        if (bi + i >=n){ 
            continue; // no more blocks to factorise => skip
        }

        auto A01 = A.submatrix(i, m, bi+i, n);
        auto A10 = A.submatrix(bi+i, m, i, i+bi);
        auto A11 = A.submatrix(bi+i, m, bi+i,n);

        auto L00 = A.submatrix(i, bi+i, i,bi+i); // top bi x bi of A00
        auto U01 = A.submatrix(i, bi+i, bi+i, n); // top bi rows of A01

        if constexpr (std::is_same_v<T,double>)
        {
            // Use BLAS to solve for U01
            // L00*U01 = A01
            cblas_dtrsm(
                CblasColMajor,
                CblasLeft,
                CblasLower, //L00 is lower triangular
                CblasNoTrans,
                CblasUnit, // Unit lower triangualer, diagonal is 1
                U01.num_rows(), // num rows in U01
                U01.num_columns(), //num columns in U01
                1.0,
                &L00(0,0), // pointer to first element of L00
                L00.ldim(),
                &U01(0,0),
                U01.ldim()
            );

            // A11 <- A11 - L10*U01
            // dgemm: C = alpha*op(A)*op(B) + beta*C
            cblas_dgemm(
                CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                A10.num_rows(),
                U01.num_columns(),
                U01.num_rows(),
                -1.0, // alpha
                &A10(0,0),
                A10.ldim(),
                &U01(0,0),
                U01.ldim(),
                1.0, // beta
                &A11(0,0),
                A11.ldim()
            );
        } else if constexpr (std::is_same_v<T,float>)
        {
            // Use BLAS to solve for U01
            // L00*U01 = A01
            cblas_strsm(
                CblasColMajor,
                CblasLeft,
                CblasLower, //L00 is lower triangular
                CblasNoTrans,
                CblasUnit, // Unit lower triangualer, diagonal is 1
                U01.num_rows(), // num rows in U01
                U01.num_columns(), //num columns in U01
                1.0,
                &L00(0,0), // pointer to first element of L00
                L00.ldim(),
                &U01(0,0),
                U01.ldim()
            );

            // A11 <- A11 - L10*U01
            // dgemm: C = alpha*op(A)*op(B) + beta*C
            cblas_sgemm(
                CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                A10.num_rows(),
                U01.num_columns(),
                U01.num_rows(),
                -1.0, // alpha
                &A10(0,0),
                A10.ldim(),
                &U01(0,0),
                U01.ldim(),
                1.0, // beta
                &A11(0,0),
                A11.ldim()
            );

        }
    }
}
// sgetrf -> double, dgtrf -> float
/**
 * See the documentation of the lu function.
 *
 * Wrapper around the LU factorization from LAPACK.
 */
template <typename T>
void lu_lapack(matrixview<T> A, vectorview<int> ipiv)
{
    // TODO: implement this function
    static_assert(std::is_same_v<T,float> || std::is_same_v<T, double>, "lu_lapack: type not supported for LAPACK implementation");
    
    int m = A.num_rows();
    int n = A.num_columns();
    int lda = A.ldim();
    int min_val = std::min(m, n);
    assert(ipiv.size() == min_val);

    int info;

    if constexpr(std::is_same_v<T, float>){
        info = LAPACKE_sgetrf(
            LAPACK_COL_MAJOR,
            m,
            n,
            A.data(),
            lda,
            ipiv.data()
        );
    } else{
        info = LAPACKE_dgetrf(
            LAPACK_COL_MAJOR,
            m,
            n,
            A.data(),
            lda,
            ipiv.data()
        );
    }

    if (info < 0) {
        throw std::runtime_error("lu_lapack: argument " + std::to_string(-info) + " had an illegal value.");
    }
    if (info > 0) {
        // matrix is singular, U(info,info) = 0
        // LU is still produced, but rank deficient => rank = 0
    }   
}
}  // namespace tws

#endif