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
            auto t = alpha * A(i, k);
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
                for (int k = kk; k < std::min(kk + block_size, A.num_columns()); ++k) {
                    for (int i = ii; i < std::min(ii + block_size, A.num_rows()); ++i) {
                        double t = alpha * A(i, k);
                        for (int j = jj; j < std::min(jj + block_size, B.num_columns()); ++j) {
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
                for (int i = ii; i < std::min(ii + block_size, A.num_rows()); ++i) {
                    for (int j = jj; j < std::min(jj + block_size, B.num_columns()); ++j) {
                        double sum = 0.;
                        for (int k = kk; k < std::min(kk + block_size, A.num_columns()); ++k) {
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
    if (A.num_rows() <= 2) {
        matmul_reordered(A, B, C, alpha, beta);
        return;
    }
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
    // TODO: implement this function
    // A = 8*k, B=k*6, C += alpha*AB

    
    const int k = A.num_columns();
    const int m = A.num_rows();
    const int n = B.num_columns();

    
    for (int i=0; i<m; i+=8){
        for (int j=0; j<n; j+=6){
            double C_cop[8][6];

            if (beta == 0.) {
                // Note: we single out beta = 0, because C might be uninitialized and
                // 0 * Nan = Nan.
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 6; ++c) {
                        C_cop[r][c] = 0.;
                    }
                }
            } else {
                for (int r = 0; r < 8; ++r) {
                    for (int c = 0; c < 6; ++c) {
                        C_cop[r][c] = C(r+i, c+j) * beta;
                    }
                }
            }
            
            for (int l = 0; l < k; ++l){
                double a[8];
                for (int y=0; y < 8; ++y){
                    a[y] = A(y+i, l);
                }
                    

                for (int x = 0; x < 6; ++x){
                    const double b = B(l, x+j);
                    for (int m=0; m < 8; ++m){
                        C_cop[m][x] += alpha*a[m]*b;
                    }
                    //C[m][n] += A[m][i]*B[i][n]*alpha
                    
                }
            }
            for (int r=0; r<8; ++r){
                for (int c=0; c<6; ++c){
                    C(r+i, c+j) = C_cop[r][c];
                }
            }
        }
    }

}

}
  // namespace tws

#endif // matmul_hpp