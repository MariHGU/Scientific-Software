#include <cassert>
#include <iostream>
#include "twsmatrix.hpp"
#include "c_lapack.h"

using namespace tws;

void call_daxpy(vector<double>& a, vector<double>& b, double& alpha){
    assert(a.size() == b.size());

    int n = a.size();
    int stride = 1;

    daxpy_(&n, &alpha, a.data(), &stride, b.data(), &stride);
}

int main(){
    double alpha = 3.0;
    int n =3;

    vector<double> x(n);
    vector<double> y(n);

    randomize(x);

    print_vector(x);
    print_vector(y);

    call_daxpy(x, y, alpha);

    print_vector(y);

    return 0;
}