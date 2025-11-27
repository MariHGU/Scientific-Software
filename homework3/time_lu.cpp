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
        // add others
        std::exit(1);
    }

    std::string method = argv[1];

    std::unordered_map<std::string,std::function<void(const tws::matrixview<>&, const tws::vector<>&)>>
    mathods = {
        {"lu_v1", tws::lu_v1},
        {"lu_v2", tws::lu_v2}
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

        tws::matrix<> A(n,n);
        tws::matrix<> B(n,n);

        randomize(A);
        randomize(B);

        // Warm-up
        for (int i=0; i < 5; ++i){
            func(a, b);
        }

        int num = 20; 
        double diff = 0.0;
        for (int i = 0; i < num; ++i){
            auto t0 = std::chrono::steady_clock::now();
            func(a,b);
            auto t1 = std::chrono::steady_clock::now();
            diff += std::chrono::duration<double>(t1-t0).count();
        }
        double avg_time = diff/num;
        double gflops = compute_gflops(n, avg_time);

        std:cout << n << " " << avg_time << " " << gflops << std::endl;
        n += 8;
    }
    
    while (n < 128){
        tws::matrix<> A(n,n);
        tws::matrix<> B(n,n);

        randomize(A);
        randomize(B);

        if (n<512){
            num = 10;
        }
        else {
            num = 5;
        }

        double diff = 0.0;
        for (int i = 0; i < num; ++i){
            auto t0 = std::chrono::steady_clock::now();
            func(a, b);
            auto t1 = std::chrono::steady_clock::now();

            diff = std::chrono::duration<double>(t1-t0).count();
        }


        double diff_avg = diff / num;
        double gflops = compute_gflops(n, diff_avg);
        
        
    }
}