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
    void operator()(vector<T> const& x, vector<T>& y) const{
        assert(x.size() == y.size());

        for(decltype(x.size()) i = 0; i<x.size(); ++i){
            y[i]=x[i]/static_cast<T>(i+1);
        }
    }
};

template<typename T>
struct matvec3{
    // want parameter m
    T m;
    
    matvec3(T m_)
    : m(m_)
    {}

    void operator()(vector<T> const& x, vector<T>& y) const{
        assert(x.size() == y.size());

        for(decltype(x.size()) i = 0; i<x.size(); ++i){
            y[i]=x[i]/static_cast<T>(i+m);
        }
    }
};

int main()
{
    int n = 100;
    vector<double> b(n);
    vector<double> b_cop = b;    
    vector<double> sol(n);
    vector<double> sol_cop = sol;
    vector<double> x(n);
    vector<double> b_ex(n);
    vector<double> b_ex_cop = b_ex;
    vector<float> bf(n);
    vector<float> bf_cop = bf;
    vector<float> solf(n);
    vector<float> solf_cop = solf;
    vector<float> xf(n);
    vector<float> b_exf(n);
    vector<float> b_exf_cop = b_exf;

    // wrap in lambda
    auto matvec_float = [](vector<float> const& x, vector<float>& y){matvec1(x, y);};

    // x random between 0 and 1
    randomize(x);
    vector<double>x_cop = x;
    randomize(xf);
    vector<float>xf_cop = xf;

    // Functor instances:
    matvec2<double> matvec2_double;
    matvec2<float> matvec2_float;

    matvec(x_cop, b_cop);
    matvec1(xf_cop,bf_cop);
    matvec2_double(x,b);
    matvec2_float(xf, bf);


    b_ex = b;
    b_ex_cop = b_cop;
    b_exf = bf;
    b_exf_cop = bf_cop;


    // x zero vector
    std::fill(x.begin(), x.end(), 0.);
    cg(matvec2_double, x, b, 1.e-10, n);
    matvec2_double(x, sol);

    std::fill(x_cop.begin(), x_cop.end(), 0.);
    cg(matvec, x_cop, b_cop, 1.e-10, n);
    matvec(x, sol);

    std::fill(xf_cop.begin(), xf_cop.end(), 0.f);
    cg(matvec_float, xf_cop, bf_cop, 1.e-10, n);
    matvec_float(xf_cop, solf_cop);

    std::fill(xf.begin(), xf.end(), 0.f);
    cg(matvec2_float, xf, bf, 1.e-10, n);
    matvec2_float(xf, solf);

    std::cout << "orig error: " << norm(sol_cop - b_ex_cop) / norm(b_ex_cop)
              << std::endl;
    std::cout << "orig error: " << norm(solf_cop - b_exf_cop) / norm(b_exf_cop)
              << std::endl;

    std::cout << "relative error: " << norm(sol - b_ex) / norm(b_ex)
              << std::endl;
    std::cout << "relative error: " << norm(solf - b_exf) / norm(b_exf)
              << std::endl;

    return 0;
}
