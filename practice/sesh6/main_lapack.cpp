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
    double alpha = 1.0;
    int n =3;

    matrix<double> A(n,n);
    vector<double> x(n);
    vector<double> y(n);

    randomize(A);
    randomize(x);
    randomize(y);

    matrix<double> A_orig = A;

    //print_vector(x);
    print_vector(y);

    int ldim = A.ldim();
    int info;
    double wkopt;
    int lwork=-1;
    char side = 'L';
    char T = 'T';
    char N = 'N';
    char U = 'U';
    int inc = 1;
    //call_daxpy(x, y, alpha);
    
    vector<double> tau(n);
    dgeqrf_(&n, &n, A.data(), &ldim, tau.data(), &wkopt, &lwork, &info);

    lwork = (int)wkopt;
    vector<double> work(lwork);
    dgeqrf_(&n, &n, A.data(), &ldim, tau.data(), work.data(), &lwork, &info);

    // compute y=Qt b
    dormqr_(&side, &T, &n, &inc, &n, A.data(), &ldim, tau.data(), y.data(), &n, work.data(), &lwork, &info);

    // solve Rx = y
    dtrsm_(&side, &U, &N, &N, &n, &inc, &alpha, A.data(), &ldim, y.data(), &n);

    // Check solution:
    double beta =0.0;
    vector<double> Ax(n);

    dgemv_(&N, &n, &n, &alpha, A_orig.data(), &ldim, y.data(), &inc, &beta, Ax.data(), &inc);
    //print_vector(y);

    print_vector(Ax);

    return 0;
}