#ifndef matmul_hpp
#define matmul_hpp

#include <cassert>

#include "twsmatrix.hpp"

namespace tws {

/**
 * Calculates the matrix-matrix product C = alpha*A*B + beta*C.
 *
 * @param A m-by-k matrix
 * @param B k-by-n matrix
 * @param C m-by-n matrix
 *          On entry, the initial value of the matrix.
 *                    if beta = 0, C does not have to be initialized.
 *          On exit, alpha*A*B + beta*C
 * @param alpha real number
 * @param beta  real number
 */
inline void matmul_naive(const matrixview<> A,
                         const matrixview<> B,
                         matrixview<> C,
                         const double alpha = 1.,
                         const double beta = 0.)
{
    if (beta == 0.) {
        // Note: we single out beta = 0, because C might be uninitialized and
        // 0 * Nan = Nan.
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) *= beta;
            }
        }
    }

    // Note: this is a rewritten version of the pure naive implementation:
    // for (int i = 0; i < A.num_rows(); ++i) {
    //     for (int j = 0; j < B.num_columns(); ++j) {
    //         for (int k = 0; k < A.num_columns(); ++k) {
    //             C(i, j) += alpha * A(i, k) * B(k, j);
    //         }
    //     }
    // }
    for (int i = 0; i < A.num_rows(); ++i) {
        for (int j = 0; j < B.num_columns(); ++j) {
            double sum = 0.;
            for (int k = 0; k < A.num_columns(); ++k) {
                sum += alpha * A(i, k) * B(k, j);
            }
            C(i, j) += sum;
        }
    }
}

inline void matmul_naive_v2(const matrixview<> A,
                            const matrixview<> B,
                            matrixview<> C,
                            const double alpha = 1.,
                            const double beta = 0.)
{
    if (beta == 0.) {
        // Note: we single out beta = 0, because C might be uninitialized and
        // 0 * Nan = Nan.
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) *= beta;
            }
        }
    }

    for (int i = 0; i < A.num_rows(); ++i) {
        for (int j = 0; j < B.num_columns(); ++j) {
            double sum = 0.;
            for (int k = 0; k < A.num_columns(); ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) += alpha * sum;
        }
    }
}

inline void matmul_reordered(const matrixview<> A,
                             const matrixview<> B,
                             matrixview<> C,
                             const double alpha = 1.,
                             const double beta = 0.)
{
    if (beta == 0.) {
        // Note: we single out beta = 0, because C might be uninitialized and
        // 0 * Nan = Nan.
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) *= beta;
            }
        }
    }

    for (int k = 0; k < A.num_columns(); ++k) {
        for (int i = 0; i < A.num_rows(); ++i) {
            t = alpha * A(i, k);
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) += t * B(k, j);
            }
        }
    }
}

inline void matmul_blocks(const matrixview<> A,
                          const matrixview<> B,
                          matrixview<> C,
                          const double alpha = 1.,
                          const double beta = 0.)
{
    // Block size 128
    // kb = nb = mb
    if (beta == 0.) {
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) *= beta;
            }
        }
    }

    int block_size = 128;
    for (int kk = 0; kk < A.num_columns(); kk += block_size) {
        for (int ii = 0; ii < A.num_rows(); ii += block_size) {
            for (int jj = 0; jj < B.num_columns(); jj += block_size) {
                for (int k = kk; k < std::min(kk + block_size, A.num_columns()), ++k) {
                    for (int i = ii; i < std::min(ii + block_size, A.num_rows()), ++i) {
                        double t = alpha * A(i, k);
                        for (int j = jj; j < std::min(jj + block_size), ++j) {
                            C(i, j) += t * B(k, j);
                        }
                    }
                }
            }
        }
    }
}

inline void matmul_blocks_b(const matrixview<> A,
                            const matrixview<> B,
                            matrixview<> C,
                            const double alpha = 1.,
                            const double beta = 0.)
{
    if (beta == 0.) {
        // Note: we single out beta = 0, because C might be uninitialized and
        // 0 * Nan = Nan.
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < A.num_rows(); ++i) {
            for (int j = 0; j < B.num_columns(); ++j) {
                C(i, j) *= beta;
            }
        }
    }
    int block_size = 128;
    for (int ii = 0; ii < A.num_rows(); ii += block_size) {
        for (int jj = 0; jj < B.num_columns(); jj += block_size) {
            for (int kk = 0; kk < A.num_columns(); kk += block_size) {
                for (int i = ii; i < std::min(ii + block_size, A.num_rows()), ++i) {
                    for (int j = jj; j < std::min(jj + block_size, B.num_columns()), ++j) {
                        double sum = 0.;
                        for (int k = kk; k < std::min(kk + block_size, A.num_columns()), ++k) {
                            sum += A(i, k) * B(k, j);
                        }
                        C(i, j) += alpa * sum;
                        }
                    }
                }
            }
        }
    }

}

inline void matmul_recursive(const matrixview<> A,
                             const matrixview<> B,
                             matrixview<> C,
                             const double alpha = 1.,
                             const double beta = 0.)
{
    // TODO: implement this function
}

/**
 * High performance matmul using a kernel
 *
 * When implementing this function, you can assume that the dimensions of A, B
 * and C are multiples of 24. Use the kernel function defined in
 * matmul_kernel.hpp to perform the actual multiplication.
 */
inline void matmul_kernel(const matrixview<> A,
                          const matrixview<> B,
                          matrixview<> C,
                          const double alpha = 1.,
                          const double beta = 0.)
{
    // TODO: implement this function
}

  // namespace tws

#endif // matmul_hpp