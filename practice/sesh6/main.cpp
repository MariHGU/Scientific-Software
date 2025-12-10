#include <cassert>
#include <iostream>
#include "twsmatrix.hpp"
#include "c_add.h"
#include "fortran_add.h"

using namespace tws;

void call_fortran(const int& n, const vector<double>& x, vector<double>& y){
    assert(x.size() == y.size());

    fortran_add_(&n, x.data(), y.data());
}

int main(){
    int n = 4;
    
    vector<double> a(n);
    vector<double> b(n);

    print_vector(a);
    print_vector(b);

    //c_add(n, a.data(), b.data());
    call_fortran(n, a, b);

    print_vector(b);
}