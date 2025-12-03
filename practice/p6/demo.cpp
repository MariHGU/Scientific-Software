// This script contains all the code snippets from the demo section
#include <iostream>
#include <cmath>

template <typename scalar>
scalar hyperbolic_sine(scalar x){ return (std::exp(x)-std::exp(-x))/((scalar) 2.); }

void snippet1() {
    std::cout<<"Single precision: "<<hyperbolic_sine<float>(1.f)<<std::endl;

    std::cout<<"Double precision: "<<hyperbolic_sine<double>(1.)<<std::endl;

    std::cout<<"Single precision: "<<hyperbolic_sine(1.f)<<std::endl; // and
    std::cout<<"Double precision: "<<hyperbolic_sine(1.)<<std::endl;
}

template<typename Fun>
double finite_difference(Fun const& f, double x) {
    double h = 1.e-8;
    return (f(x + h) - f(x - h)) / (2. * h);
}

struct sinhm{
    // constructor
    sinhm(double m)
    :m_(m)
    {}
    // overloaded ()-operator
    double operator()(double x) const{
        return hyperbolic_sine(m_*x);
    }
    // parameter m
    double m_;
};

void snippet2(){
    std::cout<<"dsinh(x)/dx|x=0 = "<<finite_difference(hyperbolic_sine<double>,0.)<<std::endl;

    sinhm sinh3(3.); // Create an instance of sinhm with m equal to 3

    std::cout<<sinh3(0.)<<std::endl; // evaluates sinh(3*0)

    std::cout<<"d sinh(3*x)/d x|x=0 = "<<finite_difference(sinh3,0.)<<std::endl;

    double m = 3.0;
    auto f = [&m] (double const& x) -> double {return m*x;};
    auto g = [m] (double const& x) -> double {return m*x;};
    std::cout<<f(4.)<<std::endl; // -> prints 12.
    std::cout<<g(4.)<<std::endl; // -> prints 12.
    // What happens when we change m?
    m = 2.0;
    std::cout<<f(4.)<<std::endl; // -> prints 8.
    std::cout<<g(4.)<<std::endl; // -> prints 12.

    m = 5;
    auto sinhml = [&m] (double x) -> double {return sinh(m*x);};
    std::cout<<finite_difference(sinhml,0.)<<std::endl;
    m=3;
    std::cout<<finite_difference(sinhml,0.)<<std::endl;
}

int main() {

    snippet1();
    snippet2();

    return 0;
}