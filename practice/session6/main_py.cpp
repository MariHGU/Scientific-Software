#include <Python.h>
#include <iostream>
#include <cassert>
#include "twsmatrix.hpp"
#include "py_add.h"

using namespace tws;

void call_py_add(vector<double>& x, vector<double>& y){
    assert(x.size() == y.size());

    int n = x.size();

    py_add(&n, x.data(), y.data());
}

int main(){

    vector<double> x(3);
    vector<double> y(3);

    randomize(x);
    randomize(y);

    print_vector(x);
    print_vector(y);

    call_py_add(x, y);

    print_vector(y);
}