#ifndef matmul_hpp
#define matmul_hpp

#include <cassert>

#include "matrix.hpp"
#include "vector.hpp"

namespace tws {

// Blocked matrix-matrix multiplication
// You may substitute this for your own implementation from homework 1
template <typename T>
void matmul_blocked_a(const matrixview<T> A,
                      const matrixview<T> B,
                      matrixview<T> C,
                      const T alpha = 1.,
                      const T beta = 0.)
{
    assert(A.num_columns() == B.num_rows());
    assert(A.num_rows() == C.num_rows());
    assert(B.num_columns() == C.num_columns());

    const int nb = 128;

    if (beta == 0) {
        for (int j = 0; j < B.num_columns(); ++j) {
            for (int i = 0; i < A.num_rows(); ++i) {
                C(i, j) = (T)0.;
            }
        }
    }
    else {
        for (int j = 0; j < B.num_columns(); ++j) {
            for (int i = 0; i < A.num_rows(); ++i) {
                C(i, j) *= beta;
            }
        }
    }

    for (int j = 0; j < B.num_columns(); j += nb) {
        for (int k = 0; k < A.num_columns(); k += nb) {
            for (int i = 0; i < A.num_rows(); i += nb) {
                int j2 = std::min(j + nb, B.num_columns());
                int k2 = std::min(k + nb, A.num_columns());
                int i2 = std::min(i + nb, A.num_rows());
                for (int jj = j; jj < j2; ++jj) {
                    for (int kk = k; kk < k2; ++kk) {
                        for (int ii = i; ii < i2; ++ii) {
                            C(ii, jj) += alpha * A(ii, kk) * B(kk, jj);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace tws

#endif