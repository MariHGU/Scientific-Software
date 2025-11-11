#include <iostream>

#include "twsmatrix.hpp"
#include "matmul.hpp"
#include "cmath"

double compute_gflops(double n, double time){
    double gflops = 0.0;
    gflops = (2*n*n*n) /(time*1e9);
    return gflops;
}


int main() {
    // TODO: timing of the different matmul
    double alpha = 1.0;
    double beta = 0.;

    
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
        tws::matrix<> A(n,n);
        tws::matrix<> B(n,n);
        tws::matrix<> C(n,n);
    
        randomize(A);
        randomize(B);
        randomize(C);

        //Warm-up
        func->second(A,B,C, alpha, beta);

        double diff = 0.0;

        for (int i = 0; i < 3; i++)
        {
            auto t0 = std::chrono::steady_clock::now();
            func->second(A,B, C, alpha, beta);
            auto t1 = std::chrono::steady_clock::now();
            diff += std::chrono::duration<double>(t1-t0).count();            
        }

        double diff_avg = diff/3;
        

        //compute gflops
        double gflops = compute_gflops(n, diff_avg);
        std::cout << n << " " << diff_avg << " " << gflops << std::endl;
        
        n += 24;
    }
    while (n <= 2400)
    {
        tws::matrix<> A(n,n);
        tws::matrix<> B(n,n);
        tws::matrix<> C(n,n);
    
        randomize(A);
        randomize(B);
        randomize(C);

        //Warm-up
        func->second(A,B,C, alpha, beta);
        
        double diff = 0.0;

        for (int i = 0; i < 3; i++)
        {
            auto t0 = std::chrono::steady_clock::now();
            func->second(A,B, C, alpha, beta);
            auto t1 = std::chrono::steady_clock::now();
            diff += std::chrono::duration<double>(t1-t0).count();            
        }

        double diff_avg = diff/3;
        

        //compute gflops
        double gflops = compute_gflops(n, diff_avg);
        std::cout << n << " " << diff_avg << " " << gflops << std::endl;
        n += 120;
    }
    return 0;
}