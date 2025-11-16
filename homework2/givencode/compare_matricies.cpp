#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "twsmatrix.hpp"  
#include "matmul.hpp"  
#include "matmul_kernel.hpp"    

using namespace tws;

template <Matrix M1, Matrix M2>
bool matrices_equal(const M1& A, const M2& B, double tol = 1e-9,
                    int* bad_i = nullptr, int* bad_j = nullptr,
                    double* va = nullptr, double* vb = nullptr)
{
    if (A.num_rows() != B.num_rows() || A.num_columns() != B.num_columns()) {
        if (bad_i) *bad_i = -1;
        if (bad_j) *bad_j = -1;
        return false;
    }
    for (int i = 0; i < A.num_rows(); ++i) {
        for (int j = 0; j < A.num_columns(); ++j) {
            double a = A(i, j);
            double b = B(i, j);
            if (std::fabs(a - b) > tol) {
                if (bad_i) *bad_i = i;
                if (bad_j) *bad_j = j;
                if (va) *va = a;
                if (vb) *vb = b;
                return false;
            }
        }
    }
    return true;
}

int main() {
    const int M = 8, N = 6, K = 24;

    matrix<> A(M, K);
    matrix<> B(K, N);
    randomize(A);
    randomize(B);

    const matrix<> A2 = A;
    const matrix<> B2 = B;

    matrix<> C_ref(M, N);
    matmul_naive(A2, B2, C_ref, 1.0, 0.0);

    // List of implementations to test
    using MM = std::function<void(const matrix<>&, const matrix<>&, matrix<>&, double, double)>;
    std::vector<std::pair<std::string, MM>> tests = {
        {"matmul_naive",        [](auto& A, auto& B, auto& C, double a, double b){ matmul_naive(A,B,C,a,b); }},
        {"matmul_naive_v2",     [](auto& A, auto& B, auto& C, double a, double b){ matmul_naive_v2(A,B,C,a,b); }},
        {"matmul_reordered",    [](auto& A, auto& B, auto& C, double a, double b){ matmul_reordered(A,B,C,a,b); }},
        {"matmul_blocks",       [](auto& A, auto& B, auto& C, double a, double b){ matmul_blocks(A,B,C,a,b); }},
        {"matmul_blocks_b",     [](auto& A, auto& B, auto& C, double a, double b){ matmul_blocks_b(A,B,C,a,b); }},
        {"matmul_recursive",    [](auto& A, auto& B, auto& C, double a, double b){ matmul_recursive(A,B,C,a,b); }},
        {"matmul_kernel",       [](auto& A, auto& B, auto& C, double a, double b){ matmul_kernel(A,B,C,a,b); }},
    };

    // Run each implementation into its own C and compare to reference
    const double alpha = 1.0, beta = 0.0;
    const double tol = 1e-9;

    std::cout << std::fixed << std::setprecision(6);
    for (auto& [name, fn] : tests) {
        matrix<> C(M, N);  // fresh destination

        auto t0 = std::chrono::high_resolution_clock::now();
        fn(A2, B2, C, alpha, beta);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int bi = -1, bj = -1; double va = 0, vb = 0;
        bool ok = matrices_equal(C, C_ref, tol, &bi, &bj, &va, &vb);

        std::cout << std::left << std::setw(20) << name
                  << (ok ? " : PASS " : " : FAIL ")
                  << " | time = " << std::setw(8) << ms << " ms";
        if (!ok) {
            std::cout << " | first diff at (" << bi << "," << bj
                      << ") got " << va << " ref " << vb;
        }
        std::cout << "\n";
    }

    return 0;
}
