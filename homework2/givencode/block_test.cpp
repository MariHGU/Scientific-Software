#include <iostream>

#include <cassert>
#include <list>

#include "twsmatrix.hpp"

namespace tws{

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
                          const double beta = 0.,
                          const int block_size = 128)
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
                            const double beta = 0.,
                            const int block_size = 128)
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


void time_all(const matrixview<> A,
                            const matrixview<> B,
                            matrixview<> C,
                            const double alpha = 1.,
                            const double beta = 0.){
    std::list<int> block_sizes = {25, 100, 500};

    auto t0_n =std::chrono::steady_clock::now();
    matmul_naive(A, B,C, alpha, beta);
    auto t1_n = std::chrono::steady_clock::now();

    double diff_n = std::chrono::duration<double>(t1_n-t0_n).count();
    
    auto t0_r =std::chrono::steady_clock::now();
    matmul_reordered(A, B,C, alpha, beta);
    auto t1_r = std::chrono::steady_clock::now();

    double diff_r = std::chrono::duration<double>(t1_r-t0_r).count();
    std::cout << "matmul_naive: " << diff_n << std::endl;
    std::cout << "matmul_recursive: " << diff_r << std::endl;
    
    
    std::cout << "bs  " << "matmul_blocks  " << "matmul_blocks_b" << std::endl;
    for (int block_size : block_sizes)
    {
        double diff=0;
        double diff_b=0;

        auto t0 = std::chrono::steady_clock::now();
        matmul_blocks(A,B, C, alpha, beta, block_size);
        auto t1 = std::chrono::steady_clock::now();
        diff += std::chrono::duration<double>(t1-t0).count();            

        auto t0_b = std::chrono::steady_clock::now();
        matmul_blocks_b(A,B, C, alpha, beta, block_size);
        auto t1_b = std::chrono::steady_clock::now();
        diff_b += std::chrono::duration<double>(t1_b-t0_b).count();  
        

        double diff_avg = diff;
        double diff_b_avg = diff_b;

        std::cout << block_size << " " << diff_avg << " " << diff_b_avg << std::endl;
    }
}
}

int main(){
    int n = 2000;
        
    double alpha = 1.0;
    double beta = 0.;
    
    tws::matrix<> A(n,n);
    tws::matrix<> B(n,n);
    tws::matrix<> C(n,n);
    
    randomize(A);
    randomize(B);
    randomize(C);
    
    time_all(A,B,C,alpha,beta);
    
    return 0;

}
