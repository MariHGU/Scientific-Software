#include <iostream>
#include "fortran_add.h"
#include "twsmatrix.hpp"


using namespace tws;

void vector_assert(const int& n, const vector<double>& a, vector<double>& b){
    assert(a.size() == b.size());

    fortran_add_(&n, a.data(), b.data());
}

int main(){
    int n = 3;

    vector<double> A(n);
    vector<double> B(n);

    randomize(A);
    randomize(B);
    
    print_vector(A);
    print_vector(B);

    vector_assert(n, A, B);

    print_vector(B);

    return 0;
}