#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

#include "twsmatrix.hpp"
#include "lu.hpp"
#include "triangular_solve.hpp"

int main(){

    tws::matrix<double> A(3,3);
    tws::matrix<double> B(3,3);
    tws::matrix<double> C(3,3);
    tws::vector<int> ipiv(3);

    randomize(A);
    randomize(B);
    randomize(C);

    // Make sure the lu factorizations compile and run
    tws::lu_v1<double>(A, ipiv);
    tws::lu_v2<double>(A, ipiv);
    tws::lu_v3<double>(A, ipiv);
    tws::lu_v4<double>(A, ipiv);
    tws::lu_lapack<double>(A, ipiv);
    std::cout << "lu factorization ran succesfully" << std::endl;

    // Make sure the triangular solves compile and run
    tws::trsm_ll_v1<double>(A, B);
    tws::trsm_ll_v2<double>(A, B);
    std::cout << "triangular solve ran succesfully" << std::endl;
}