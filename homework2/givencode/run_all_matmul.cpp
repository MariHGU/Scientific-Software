#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "twsmatrix.hpp"
#include "matmul.hpp"

int main(){

    tws::matrix<> A(24,24);
    tws::matrix<> B(24,24);
    tws::matrix<> C(24,24);

    randomize(A);
    randomize(B);
    randomize(C);

    const tws::matrix<> A2 = A;
    const tws::matrix<> B2 = B;

    // Run all matrix-matrix multiplication implementations
    // but do not check anything.
    tws::matmul_naive(A2, B2, C, 1.0, 0.0);
    tws::matmul_naive_v2(A2, B2, C, 1.0, 0.0);
    tws::matmul_reordered(A2, B2, C, 1.0, 0.0);
    tws::matmul_blocks(A2, B2, C, 1.0, 0.0);
    tws::matmul_blocks_b(A2, B2, C, 1.0, 0.0);
    tws::matmul_recursive(A2, B2, C, 1.0, 0.0);
    tws::matmul_kernel(A2, B2, C, 1.0, 0.0);
    std::cout << "matrix-matrix multiplication ran succesfully" << std::endl;
}