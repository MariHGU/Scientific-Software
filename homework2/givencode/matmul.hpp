#ifndef matmul_hpp
#define matmul_hpp

#include <cassert>

#include "twsmatrix.hpp"
#include "matmul_kernel.hpp"

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
    int m = A.num_rows();
    int n = B.num_columns();
    int K = A.num_columns();

    if (beta == 0.) {
        // Note: we single out beta = 0, because C might be uninitialized and
        // 0 * Nan = Nan.
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                C(i, j) *= beta;
            }
        }
    }

    for (int j = 0; j < n; ++j)
    {
        for (int k = 0; k < K; ++k)
        {
            auto t = alpha*B(k, j);
            for (int i=0; i < m; ++i)
            {
                C(i, j) += t * A(i, k);
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

    int M = A.num_rows();
    int N = B.num_columns();
    int K = A.num_columns();

    if (beta == 0.) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C(i, j) *= beta;
            }
        }
    }

    int block_size = 128;

    for (int jj = 0; jj < N; jj += block_size) {
        for (int kk = 0; kk < K; kk += block_size) {
            for (int ii = 0; ii < M; ii += block_size) {

                for (int j = jj; j < std::min(jj + block_size, N); ++j) {
                    for (int k = kk; k < std::min(kk + block_size, K); ++k) {
                        double t = alpha * B(k, j);
                        for (int i = ii; i < std::min(ii + block_size, M); ++i) {
                            C(i, j) += t * A(i, k);
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
    int M = A.num_rows();
    int N = B.num_columns();
    int K = A.num_columns();


    if (beta == 0.) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C(i, j) = 0.;
            }
        }
    } else {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C(i, j) *= beta;
            }
        }
    }
    int block_size = 128;

    for (int ii = 0; ii < M; ii += block_size) {
        for (int jj = 0; jj < N; jj += block_size) {
            for (int kk = 0; kk < K; kk += block_size) {

                for (int i = ii; i < std::min(ii + block_size, M); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, N); ++j) {
                        double sum = 0.;
                        for (int k = kk; k < std::min(kk + block_size, K); ++k) {
                            sum += A(i, k) * B(k, j);
                        }
                        C(i, j) += alpha * sum;
                        }
                    }
                }
            }
        }
    }



static inline std::tuple<tws::matrixview<>, tws::matrixview<>, tws::matrixview<>, tws::matrixview<>> split(const tws::matrixview<> M){
    const int col_half = M.num_columns() / 2;
    const int row_half = M.num_rows() / 2;

    tws::matrixview<> a = M.submatrix(0, row_half, 0, col_half);
    tws::matrixview<> b = M.submatrix(0, row_half, col_half, M.num_columns());
    tws::matrixview<> c = M.submatrix(row_half, M.num_rows(), 0, col_half);
    tws::matrixview<> d = M.submatrix(row_half, M.num_rows(), col_half, M.num_columns());
    return {a, b, c, d};
}

inline void matmul_recursive(const matrixview<> A,
                             const matrixview<> B,
                             matrixview<> C,
                             const double alpha = 1.,
                             const double beta = 0.)
{
    if (A.num_rows() <= 96) {
        matmul_reordered(A, B, C, alpha, beta);
        return;
    }

    // Recursive multiplication until size is small enough

    // c00 = ae+bg, c01 = af + bh, c10 = ce + dg, c11 = cf + dh
    auto [a, b, c, d] = split(A);
    auto [e, f, g, h] = split(B);
    auto [c00, c01, c10, c11] = split(C);

    matmul_recursive(a, e, c00, alpha, beta);
    matmul_recursive(b, g, c00, alpha, 1.);

    matmul_recursive(a, f, c01, alpha, beta);
    matmul_recursive(b, h, c01, alpha, 1.);

    matmul_recursive(c, e, c10, alpha, beta);
    matmul_recursive(d, g, c10, alpha, 1.);

    matmul_recursive(c, f, c11, alpha, beta);
    matmul_recursive(d, h, c11, alpha, 1.);
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
    // A = 8*k, B=k*6, C += alpha*AB
    const int M = A.num_rows();
    const int N = B.num_columns();
    const int K = A.num_columns(); 

    assert(M % 8 == 0);
    assert(N % 6 == 0);
    assert(K > 0);

    if (beta != 1.0) {
        if (beta == 0.0) {
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < M; ++i)
                    C(i, j) = 0.0;
        } else {
            for (int j = 0; j < N; ++j)
                for (int i = 0; i < M; ++i)
                    C(i, j) *= beta;
        }
    }

    for (int j = 0; j < N; j += 6) {  
        for (int i = 0; i < M; i += 8) {  

            double* C_block = &C(i, j);
            int ldc = C.leading_dimension();  // move between columns

            const double* A_block = &A(i, 0);
            int lda = A.leading_dimension();  // >= M

            const double* B_block = &B(0, j);
            int ldb = B.leading_dimension();  // >= K

            kernel_8x6(K, alpha, A_block, lda, B_block, ldb, C_block, ldc);
        }
    }
}

}
  // namespace tws

#endif // matmul_hpp