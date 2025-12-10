#include <cassert>
#include <iostream>
#include "twsmatrix.hpp"
#include "c_add.h"

using namespace tws;

int main(){
    int n = 4;
    
    vector<double> a(n);
    vector<double> b(n);

    print_vector(a);
    print_vector(b);

    c_add(n, a.data(), b.data());

    print_vector(b);
}