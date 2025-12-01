#include <iostream>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <cstdlib> // for std::exit

#include "twsmatrix.hpp"
#include "matmul.hpp"
#include "cmath"

double compute_gflops(double n, double time){
    return (2.0 * n * n * n)/(time * 1e9);
}

int main(int argc, char* argv[]){
    if (argc != 2){
        std::cerr << "Usage: " << argv[0] << " <matmul_method>\n";
        std::cerr << "Available methods:\n";
        std::cerr << " lu_v1\n";
        std::cerr << " lu_v2\n";
        std::cerr << " lu_v3\n";
        std::cerr << " lu_v4\n";
        std::cerr << " lu_lapack\n";
        std::exit(1);
    }

    std::string method = argv[1];

    std::unordered_map<std::string,std::function<void(tws::matrixview<double>&, tws::vectorview<int>&)>>
    methods = {
        {"lu_v1", tws::lu_v1},
        {"lu_v2", tws::lu_v2},
        {"lu_v3", tws::lu_v3},
        {"lu_v4", tws::lu_v4},
        {"lu_lapack", tws::lu_lapack}
    };

    auto& func_it = methods.find(method);
    if (func_it == methods.end()){
        std::cerr << "Error: unknown method " << method << "\n";
        std::exit(1);
    }

    auto& func = func_it->second;

    std::cout << "n  time  gflops" << std::endl;

    int n=8;
    while (n < 128){

        tws::matrix<> A0(n,n);
        randomize(A0);
        
        
        // Warm-up
        for (int i=0; i < 5; ++i){
            tws::matrix<double> A = A0;
            tws::vector<> ipiv(n);
            func(matrixview<double>(A), vectorview<int>(ipiv));
        }

        int num = 20; 
        double diff = 0.0;
        for (int i = 0; i < num; ++i){
            tws::matrix<double A = A0;
            tws::vector<int> ipiv(n);

            auto t0 = std::chrono::steady_clock::now();
            func(matrixview<double>(A),matrixview<double>(ipiv));
            auto t1 = std::chrono::steady_clock::now();

            diff += std::chrono::duration<double>(t1-t0).count();
        }
        double avg_time = diff/num;
        double gflops = compute_gflops(n, avg_time);

        std:cout << n << " " << avg_time << " " << gflops << std::endl;
        n += 8;
    }
    
    while (n < 2048){
        tws::matrix<> A0(n,n);
        randomize(A0);

        // Warm-up
        for (int i=0; i < 3; ++i){
            tws::matrix<double> WU = A0;
            tws::vector<int> ipiv_wu(n);
            func(matrixview<double>(WU), vectorview<int>(ipiv_wu));
        }
        
        if (n<512){
            num = 10;
        }
        else {
            num = 5;
        }
        
        double diff = 0.0;
        for (int i = 0; i < num; ++i){
            tws::matrix<double> A = A0; // copy to ensure every test is run on the same data
            tws::matrix<int> ipiv(n);

            auto t0 = std::chrono::steady_clock::now();
            func(matrixview<double>(A), vectorview<int>(ipiv));
            auto t1 = std::chrono::steady_clock::now();

            diff = std::chrono::duration<double>(t1-t0).count();
        }


        double diff_avg = diff / num;
        double gflops = compute_gflops(n, diff_avg);

        n += 128;
    }
}