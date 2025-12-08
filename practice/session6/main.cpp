#include <iostream>
#include "c_add.h"
#include "twsmatrix.hpp"


using namespace tws;

int main(){
    vector<double> A(3);
    vector<double> B(3);

    randomize(A);
    randomize(B);

    print_vector(A);
    print_vector(B);

    c_add(3, A.data(), B.data());

    print_vector(A);

    return 0;

}