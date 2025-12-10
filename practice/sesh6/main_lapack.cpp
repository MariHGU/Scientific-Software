#include <cassert>
#include <iostream>
#include "twsmatrix.hpp"
#include "c_lapack.h"

using namespace tws;

void call_daxpy(vector<double>& a, vector<double>& b, double& alpha){
    assert(a.size() == b.size());

    int n = a.size()

    daxpy_(&n, &alpha, a.data(), 1, b.data(), 1);
}

int main(){
    double alpha = 3.0;
    int n =3;

    vector<double> x(n);
    vector<double> y(n);

    randomize(x);

    call_daxpy(x, y, alpha);

    return 0;
}