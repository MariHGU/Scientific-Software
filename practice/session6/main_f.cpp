#include <iostream>
#include "fortran_add.h"
#include "twsmatrix.hpp"


using namespace tws;

int main(){
    vector<double> A(3);
    vector<double> B(3);

    randomize(A);
    randomize(B);

    print_vector(A);
    print_vector(B);

    fortran_add_(3, A.data(), B.data());

    print_vector(B);

    return 0;

}