#ifndef triangular_solve_hpp
#define triangular_solve_hpp

#include <cassert>

#include "matrix.hpp"
#include "matmul.hpp"

namespace tws {

/**
 * Solves the triangular system L*X = B, where L is a unit lower triangular
 * matrix. The solution X is stored in B.
 * 
 * Initial version of the triangular solve provided by the assistants.
 * NOTE: do not change this function.
 */
template <typename T>
void trsm_ll_v1(const matrixview<T> L, matrixview<T> B)
{
    int n = L.num_rows();
    int m = B.num_columns();

    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < n; ++k) {
            // Note: we assume that the diagonal of L is equal to 1.
            // so we don't need to divide by L(j, j).
            // B(k, j) = B(k, j) / L(k, k);
            for (int i = k + 1; i < n; ++i) {
                B(i, j) -= L(i, k) * B(k, j);
            }
        }
    }
}

/**
 * Solves the triangular system L*X = B, where L is a unit lower triangular
 * matrix. The solution X is stored in B.
 * 
 * Optimized version of the triangular solve using blocking.
 */
template <typename T>
void trsm_ll_v2(const matrixview<T> L, matrixview<T> B)
{
    // TODO: implement this function
    int n = L.num_rows();
    int m = B.num_columns();
    int nb = 64;
    // want to max block size to fit within L1 cache, L2 cache and L3 cache

    // iterate through blocks
    for (int k = 0; k < n; k += nb){
        int bk = std::min(nb, n-k); // find the index of first item in block

        // Find diagonal blocks
        auto Lkk = L.submatrix(k, bk, k, bk); // square (bi-i x bi-i matrix)
        auto Bk = B.submatrix(k, bk, 0, m); // (bk-i x m matrix)

        // X00 = L00^-1*B00
        trsm_ll_v1(Lkk, Bk); // solution of x is now stored in Bk

        int i_next = bk + k; // index of start of next block
        if (i_next < n){
            auto Lik = L.submatrix(i_next, n-i_next, k, bk); // (n-2i_inext x bk-k matrix)
            auto Bi = B.submatrix(i_next, n-i_next, 0, m); 

            // Bi = Bi - Lik*Xk
            matmul_blocked_a(Lik, Bk, Bi, T(-1), T(0)); // -> stores solution in Bi
            // next iteration will solve X10 = L11^-1(B10 - L10*L00^-1*B00)
        }
    }
}

}  // namespace tws

#endif