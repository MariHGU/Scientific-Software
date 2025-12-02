#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <functional>

#include "twsmatrix.hpp"
#include "matmul.hpp"
#include "lu.hpp"

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
                double a = A(i, j); // faster for culumn iteration
                double b = B(i, j);
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

template <typename T>
void extract_LU(const matrixview<T>& A_lu,
                matrix<T>& L,
                matrix<T>& U)
{
    int m = A_lu.num_rows();
    int n = A_lu.num_columns();
    int k = std::min(m, n);

    assert(L.num_rows() == m && L.num_columns() == k);
    assert(U.num_rows() == k && U.num_columns() == n);

    // L: m x k (unit lower)
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < m; ++i) {
            if (i > j) {
                L(i,j) = A_lu(i,j);
            } else if (i == j) {
                L(i,j) = T(1); // unit diagonal
            } else {
                L(i,j) = T(0);
            }
        }
    }

    // U: k x n (upper)
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < k; ++i) {
            if (i <= j) {
                U(i,j) = A_lu(i,j);
            } else {
                U(i,j) = T(0);
            }
        }
    }
}

template <typename T>
void apply_pivots(matrix<T>& A_orig, const vector<int>& ipiv){
    int m = A_orig.num_rows();
    int n = A_orig.num_columns();
    int k = ipiv.size();

    assert(k == std::min(m,n));
    for (int i = 0; i < k; ++i){
        int piv = ipiv[i];
        if (piv == i){
            continue;
        }
        for (int col = 0; col < n; ++col){
            std::swap(A_orig(i, col), A_orig(piv, col));
        }
    }
}

template <typename T>
double frob_norm(const matrix<T>& A){
    double sum = 0.0;
    int m= A.num_rows();
    int n = A.num_columns();

    for (int j = 0; j < n; ++j){
        for (int i = 0; i < m; ++i){
            double val = static_cast<double>(A(i, j));
            sum += val*val;
        }
    }
    return std::sqrt(sum);
}

template <typename T>
bool lu_residual(const matrix<T>& A_orig, matrix<T>& A_lu, vector<int>& ipiv){
    int m = A_orig.num_rows();
    int n = A_orig.num_columns();
    int minval = std::min(m,n);

    double tol = std::is_same_v<T, float> ? 1e-5 : 1e-12; // float has at most 1e-7 precision, and double 1e-15

    // reconstruct L and U from A_lu
    matrix<T> L(m, minval), U(minval, n);
    extract_LU(matrixview<T>(A_lu), L, U);

    // Apply pivoting to A_orig to get P*A_orig
    matrix<T> PA = A_orig;
    apply_pivots(PA, ipiv);

    // Compute residual R = PA - L*
    matrix<T> LU(m, n);
    matmul_blocked_a(matrixview(L), matrixview(U), matrixview(LU), T(1.0), T(0.0));

    for (int j = 0; j < n; ++j){
        for (int i=0; i < m; ++i)
        {
            LU(i, j) = PA(i, j) - LU(i, j);
        }
        
    }

    double norm_PA = frob_norm(PA);
    double norm_R = frob_norm(LU);

    return norm_R / norm_PA < tol;
}    

// implement further tests??
// test non-square
// test wrong 
// check if reconstructed A is close to original A
// zero-size cases
using LU = std::function<void(matrixview<double>, vectorview<int>)>;

bool non_square_residual_test(LU lu_func) {
    int m = 5;
    int n = 10;
    int k = std::min(m, n);

    // Original matrix
    matrix<double> A0(m, n);
    randomize(A0);

    // Copy to factorize
    matrix<double> A = A0;
    vector<int> ipiv(k);

    // Run LU
    lu_func(matrixview<double>(A), vectorview<int>(ipiv));

    // Compute and test relative residual ||P A0 - L U|| / ||A0||
    bool rel_res = lu_residual(A0, A, ipiv);

    return rel_res;
}

bool test_wrong_size_handling(LU lu_func){
    // test too small ipiv:
    int m = 5;
    int n = 5;
    matrix<double> A(m, n);
    randomize(A);
    vector<int> ipiv(3); // should be min(m,n)=5

    bool error_thrown = false;

    try{
        lu_func(matrixview<double>(A), vectorview<int>(ipiv));
    } catch (const std::exception& e){
        error_thrown = true; // correctly threw error
    }

    // test too large ipiv:
    vector<int> ipiv(10);
    try{
        lu_func(matrixview<double>(A), vectorview<int>(ipiv));
    } catch (const std::exception& e){
        error_thrown = true; // correctly threw error
    }
    return error_thrown; // no error thrown
}

bool zero_size_test(LU lu_func){
    // test zero size matrix
    matrix<double> A(0,0);
    vector<int> ipiv(0);

    bool error_thrown = false;

    try{
        lu_func(matrixview<double>(A), vectorview<int>(ipiv));
    } catch (const std::exception& e){
        error_thrown = true; // should not throw error
    }

    return !error_thrown; // no error thrown
}

bool test_float_handling(LU lu_func){
    int m = 5;
    int n = 5;
    matrix<float> A(m,n);
    randomize(A);
    matrix<float> A_orig = A;
    vector<int> ipiv(std::min(m,n));

    bool error_thrown = false;
    try
    {
        lu_func(matrixview<float>(A), vectorview<int>(ipiv));
    }
    catch(const std::exception& e)
    {
        error_thrown = true; // should not throw error
    }

    if (!error_thrown){
        // compute residual
        bool rel_res = lu_residual(A_orig, A, ipiv);
        return rel_res;
    }
    return false;   
}


int main(){
    // init matricies
    int M = 64;
    int N = 64;
    matrix<double> A_orig(M, N);
    randomize(A_orig);

    matrix<double> A_ref = A_orig;
    vector<int> ipiv_ref(std::min(M,N));
    lu_v1(matrixview(A_ref), vectorview(ipiv_ref)); // reference solution

    // List of implementations to test
    std::vector<std::pair<std::string, LU>> tests = {
        {"lu_v1",   [](matrixview<double> A, vectorview<int> ipiv) { lu_v1(A, ipiv); }},
        {"lu_v2",   [](matrixview<double> A, vectorview<int> ipiv) { lu_v2(A, ipiv); }},
        {"lu_v3",   [](matrixview<double> A, vectorview<int> ipiv) { lu_v3(A, ipiv); }},
        {"lu_v4",   [](matrixview<double> A, vectorview<int> ipiv) { lu_v4(A, ipiv); }},
        {"lu_lapack",   [](matrixview<double> A, vectorview<int> ipiv) { lu_lapack(A, ipiv); }}
    };

    const double tol = 1e-9;

    std::cout << std::fixed << std::setprecision(9);
    for (auto& [name, fn] : tests){
        // test A
        matrix<double> A = A_orig;
        vector<int> ipiv(std::min(M,N));
        auto t0 = std::chrono::steady_clock::now();
        fn(matrixview(A), vectorview(ipiv));
        auto t1 = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double>(t1-t0).count();

        int bi = -1, bj = -1; double va = 0, vb = 0;
        bool pass_1 = matricies_equal(A, A_ref, tol, &bi, &bj, &va, &vb);

        std::cout << std::left << std::setw(12) << name
                 << (pass_1 ? " : PASS " : " : FAIL ")
                 << " | time = " << std::setw(8) << time << "s";
        if(!pass_1){
            std::cout << " | first difference at (" << bi <<"," << bj
                      << ") got " << va << " ref " << vb;
        }
        std::cout << "\n";

        // test residual:
        bool pass_2 = lu_residual(A_orig, A, ipiv);
        std::cout << "Test residual: " << (pass_2 ? "PASS" : "FAIL") << "\n";

        bool pass_3 = non_square_residual_test(fn);
        std::cout << "Test non-square: " << (pass_3 ? "PASS" : "FAIL") << "\n";

        bool pass_4 = test_wrong_size_handling(fn);
        std::cout << "Test wrong size handling: " << (pass_4 ? "PASS" : "FAIL") << "\n";

        bool pass_5 = zero_size_test(fn);
        std::cout << "Test zero size: " << (pass_5 ? "PASS" : "FAIL") << "\n";

        bool pass_6 = test_float_handling(fn);
        std::cout << "Test float handling: " << (pass_6 ? "PASS" : "FAIL") << "\n";
    }
    return 0;
}