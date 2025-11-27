#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>

#include "twsmatrix.hpp"
#include "matmul.hpp"
#include "matmul_kernel.hpp"

using namespace tws;

template <Matrix M1, Matrix M2>
bool matricies_equal(const M1& A, const M2& B, double tol = 1e-9,
                    int* bad_i = nullptr, int* bad_j = nullptr,
                    double* va = nullptr, double* vb = nullptr)
    {
        if (A.num_rows() != B.num_rows() || A.num_columns() != B.num_columns()){
            if (bad_i) *bad_i = -1;
            if (bad_j) *bad_j = -1;
            // -1 => error in size
            return false;
        }

        for(int i = 0; i < A.num_rows(); ++i){
            for (int j=0; j<A.num_columns(); ++j){
                double a = A(j, i); // faster for culumn iteration
                double b = B(j, i);
                if (std::fabs(a-b) > tol){
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

// implement further tests??
// test non-square
// test wrong 

int main(){
    // init matricies
    const matrixview<> A_ref(M, N);
    const vectorview<> ipiv_ref(M);
    randomize(A_ref);
    randomize(ipipv_ref);

    const matrixview<> A = A_ref;
    const vectorview<> ipiv = ipiv_ref;

    lu_v1(A_ref, ipiv_ref);

    // List of implementations to test
    using LU = std::function<void(const matrixview<>&, const vectorview<>&);
    std::vector<std::pair<std::string, LU>> tests = {
        {"lu_v1",   [](auto& A, auto& ipiv)},
        {"lu_v2",   [](auto& A, auto& ipiv)},
        {"lu_v3",   [](auto& A, auto& ipiv)}
        // add the others
    };

    const double tol = 1e-9;

    std::cout << std::fixed << std::setprecision(9);
    for (auto& [name, fn] : tests){
        auto t0 = std::chrono::steady_clock::now();
        fn(A, ipiv);
        auto t1 = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(t1-t0).count();

        int bi = -1, bj = -1; double va = 0, vb = 0;
        bool pass = matricies_equal(A, A_ref, tol, &bi, &bj, &va, &vb);

        std::cout << std::left << std::setw << name
                 << (pass ? " : PASS " : " : FAIL ")
                 << " | time = " << std::setw(8) << time << "s";
        if(!pass){
            std::cout << " | first difference at (" << bi <<"," << bj
                      << ") got " << va << " ref " << vb;
        }
        std::cout << "\n";
    }

    return 0;



}