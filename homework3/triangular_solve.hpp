#ifndef triangular_solve_hpp
#define triangular_solve_hpp

#include <cassert>

#include "matrix.hpp"

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

    for (int j = 0; j < m; ++j){
        for (int i = 0; i < n; ++i){
            for (int k = 0; k < i; ++k){
                B(i, j) = B(i,j) - L(i,k)*B(k,j);
            }

            B(i, j) = B(i,j)/L(i,i);
        }
    }
}

}  // namespace tws

#endif