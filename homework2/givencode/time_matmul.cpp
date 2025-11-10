#include <iostream>

#include "twsmatrix.hpp"
#include "matmul.hpp"
#include "cmath"

double compute_gflops(double n, double time){
    double gflops = 0.0;
    gflops = 2*pow(n, 3) /((time/1000)*1e9);
    return gflops;
}


int main() {
    // TODO: timing of the different matmul
    tws::matrix<> A(24,24);
    tws::matrix<> B(24,24);
    tws::matrix<> C(24,24);

    double alpha = 1.0;
    double beta = 0.;

    randomize(A);
    randomize(B);
    randomize(C);

    const tws::matrix<> A2 = A;
    const tws::matrix<> B2 = B;
    
    std::unordered_map<std::string,
        std::function<void(const tws::matrix<>&, const tws::matrix<>&, tws::matrix<>&, double, double)>>
        methods = {
            {"matmul_naive",      tws::matmul_naive},
            {"matmul_naive_v2",   tws::matmul_naive_v2},
            {"matmul_reordered",  tws::matmul_reordered},
            {"matmul_blocks",     tws::matmul_blocks},
            {"matmul_blocks_b",   tws::matmul_blocks_b},
            {"matmul_recursive",  tws::matmul_recursive},
            {"matmul_kernel",     tws::matmul_kernel}
        };
    
    std::string method;
    std::cout << "Method name: ";
    std::cin >> method;

    auto func = methods.find(method);
    int n = 0;
    std::cout << "n  " << "time  " << "gflops" << std::endl;



    while (n < 120)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        func->second(A,B, C, alpha, beta);
        auto t1 = std::chrono::high_resolution_clock::now();
        double diff = std::chrono::duration<double, std::milli>(t1-t0).count();

        //compute gflops
        double gflops = compute_gflops(n, diff);
        std::cout << n << " " << diff << " " << gflops << std::endl;
        
        n += 24;
    }
    while (n <= 2400)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        func->second(A,B, C, alpha, beta);
        auto t1 = std::chrono::high_resolution_clock::now();
        double diff = std::chrono::duration<double, std::milli>(t1-t0).count();

        //compute gflops
        double gflops = compute_gflops(n, diff);
        std::cout << n << " " << diff << " " << gflops << std::endl;
        n += 120;
    }
    return 0;
}