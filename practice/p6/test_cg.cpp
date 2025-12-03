#include <algorithm>
#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "cg.hpp"
#include "twsmatrix.hpp"

using namespace tws;

void matvec(vector<double> const& x, vector<double>& y)
{
    assert(x.size() == y.size());

    for (decltype(x.size()) i = 0; i < x.size(); ++i) {
        y[i] = x[i] / (i + 1);
    }
}

// templated
template<typename T>
void matvec1(vector<T> const& x, vector<T>& y){
    assert(x.size() == y.size());

    for (decltype(x.size()) i = 0; i < x.size(); ++i) {
        y[i] = x[i] / static_cast<T>(i + 1);
    }
}

//functor
template<typename T>
struct matvec2{
    //costructor 
    void operator()(vector<T> const& x, vector<T>& y){
        assert(x.size() == y.size());

        for(decltype(x.size())) i = 0; i<x.size(); ++i{
            y[i]=x[i]/static_cast<T>(i+1);
        }
    }
};

int main()
{
    int n = 100;
    vector<double> b(n);
    vector<double> sol(n);
    vector<double> x(n);
    vector<double> b_ex(n);
    vector<float> bf(n);
    vector<float> solf(n);
    vector<float> xf(n);
    vector<float> b_exf(n);

    // wrap in lambda
    auto matvec_float = [](vector<float> const& x, vector<float>& y){matvec1(x, y);};

    // x random between 0 and 1
    randomize(x);
    randomize(xf);

    // Functor instances:
    matvec2<double> matvec2_double;
    matvec2<float> matvec2_float;

    //matvec(x, b);
    //matvec1(xf,bf);
    matvec2_double(x,b);
    matvec2_float(xf, bf);


    b_ex = b;
    b_exf = bf;

    // x zero vector
    std::fill(x.begin(), x.end(), 0.);
    cg(matvec2_double, x, b, 1.e-10, n);
    matvec2_double(x, sol);

    std::fill(xf.begin(), xf.end(), 0.f);
    cg(matvec2_float, xf, bf, 1.e-10, n);
    matvec2_float(xf, solf);

    std::cout << "relative error: " << norm(sol - b_ex) / norm(b_ex)
              << std::endl;
    std::cout << "relative error: " << norm(solf - b_exf) / norm(b_exf)
              << std::endl;

    return 0;
}
