#include <iostream>
#include "fortran_matvec.h"

using namespace tws;

void matvec_fortran(const int m, const int n, const matrix<double>& M, const vector<double>& x, vector<double>& y){
    assert(x.size() == n);
    assert(y.size() == m);

    int lda = M.ldim();

    fortran_matvec_(&m, &n, M.data(), &lda, x.data(), y.data());
}


int main(){
    int m = 3;
    int n = 4;

    matrix<double> M(m, n);
    vector<double> x(n);
    vector<double> y(m);

    matvec_fortran(m, n, M, x, y);

    return 0;
}