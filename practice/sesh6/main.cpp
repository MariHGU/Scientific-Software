#include <cassert>
#include <iostream>
#include "twsmatrix.hpp"
#include "c_add.h"
#include "fortran_add.h"
#include "fortran_matvec.h"

using namespace tws;

void call_fortran(const int& n, const vector<double>& x, vector<double>& y){
    assert(x.size() == y.size());

    fortran_add_(&n, x.data(), y.data());
}

void call_fortran_matvec(const int& m, const int& n, const matrix<double>& A, const vector<double>& x, vector<double> y){
    assert(m == A.num_rows());
    assert(n == A.num_columns());
    assert(x.size() == n);
    assert(y.size() == m);

    int lda = A.ldim()

    fortran_matvec_(&m, &n, A.data(), &lda, x.data(), y.data());
}

int main(){
    int n = 4;
    int m = 3;
    
    matrix<double> M(m,n);
    vector<double> a(n);
    vector<double> b(m);

    a[0] = 1, a[1] = 2, a[2] = 3; 
    randomize(M);

    print_vector(a);
    print_matrix(M);

    //c_add(n, a.data(), b.data());
    //call_fortran(n, a, b);

    call_fortran_matvec(m, n, M, a, b);

    print_vector(b);
}