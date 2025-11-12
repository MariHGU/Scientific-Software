#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>

#include "twsmatrix.hpp"
#include "matmul.hpp"


inline void matmul_blocks(const tws::matrixview<> A,
                          const tws::matrixview<> B,
                          tws::matrixview<> C,
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

inline void matmul_blocks_b(const tws::matrixview<> A,
                            const tws::matrixview<> B,
                            tws::matrixview<> C,
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


int main(int argc, char* argv[]) {
    const int N = 2000;
    double alpha = 1.0;
    double beta = 0.0;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <method> [block_size]\n";
        std::cerr << "Methods: matmul_naive, matmul_reordered, matmul_blocks, matmul_blocks_b\n";
        return 1;
    }

    std::string method = argv[1];
    int block_size = 128; // default, ignored by non-blocked methods

    if (argc >= 3) {
        block_size = std::atoi(argv[2]);
    }

    tws::matrix<> A(N, N);
    tws::matrix<> B(N, N);
    tws::matrix<> C(N, N);

    randomize(A);
    randomize(B);


    if (method == "matmul_naive") {
        auto t0 = std::chrono::steady_clock::now();
        tws::matmul_naive(A, B, C, alpha, beta);
        auto t1 = std::chrono::steady_clock::now();
        double diff = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "bs: " << block_size << ", time: " << diff << std::endl;

    }
    else if (method == "matmul_reordered") {
        auto t0 = std::chrono::steady_clock::now();
        tws::matmul_reordered(A, B, C, alpha, beta);
        auto t1 = std::chrono::steady_clock::now();
        double diff = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "bs: " << block_size << ", time: " << diff << std::endl;

    }
    else if (method == "matmul_blocks") {
        auto t0 = std::chrono::steady_clock::now();
        matmul_blocks(A, B, C, alpha, beta, block_size);
        auto t1 = std::chrono::steady_clock::now();
        double diff = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "bs: " << block_size << ", time: " << diff << std::endl;

    }
    else if (method == "matmul_blocks_b") {
        auto t0 = std::chrono::steady_clock::now();
        matmul_blocks_b(A, B, C, alpha, beta, block_size);
        auto t1 = std::chrono::steady_clock::now();
        double diff = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "bs: " << block_size << ", time: " << diff << std::endl;
    }
    else {
        std::cerr << "Unknown method: " << method << "\n";
        return 1;
    }

    return 0;
}