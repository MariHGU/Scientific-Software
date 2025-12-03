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

int main()
{
    int n = 100;
    vector<double> b(n);
    vector<double> sol(n);
    vector<double> x(n);
    vector<double> b_ex(n);

    // x random between 0 and 1
    randomize(x);

    matvec(x, b);

    b_ex = b;

    // x zero vector
    std::fill(x.begin(), x.end(), 0.);
    cg(matvec, x, b, 1.e-10, n);
    matvec(x, sol);

    std::cout << "relative error: " << norm(sol - b_ex) / norm(b_ex)
              << std::endl;

    return 0;
}
