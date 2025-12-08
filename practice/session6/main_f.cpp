#include <iostream>
#include "fortran_add.h"
#include "twsmatrix.hpp"


using namespace tws;

int main(){
    int n = 3;

    vector<double> A(n);
    vector<double> B(n);

    randomize(A);
    randomize(B);

    print_vector(A);
    print_vector(B);

    fortran_add_(&n, A.data(), B.data());

    print_vector(B);

    return 0;

}