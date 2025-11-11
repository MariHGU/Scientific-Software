#include <iostream>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <cstdlib> // for std::exit

#include "twsmatrix.hpp"
#include "matmul.hpp"
#include "cmath"

double compute_gflops(double n, double time){
    return (2.0 * n * n * n) / (time * 1e9);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matmul_method>\n";
        std::cerr << "Available methods:\n";
        std::cerr << "  matmul_naive\n";
        std::cerr << "  matmul_naive_v2\n";
        std::cerr << "  matmul_reordered\n";
        std::cerr << "  matmul_blocks\n";
        std::cerr << "  matmul_blocks_b\n";
        std::cerr << "  matmul_recursive\n";
        std::cerr << "  matmul_kernel\n";
        std::exit(1);
    }

    std::string method = argv[1];

    double alpha = 1.0;
    double beta = 0.0;

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

    auto func_it = methods.find(method);
    if (func_it == methods.end()) {
        std::cerr << "Error: unknown method '" << method << "'\n";
        std::exit(1);
    }

    auto& func = func_it->second;

    std::cout << "n  time  gflops" << std::endl;

    int n = 0;
    while (n < 120) {
        tws::matrix<> A(n, n);
        tws::matrix<> B(n, n);
        tws::matrix<> C(n, n);

        randomize(A);
        randomize(B);
        randomize(C);

        // Warm-up
        func(A, B, C, alpha, beta);

        double diff = 0.0;
        for (int i = 0; i < 3; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            func(A, B, C, alpha, beta);
            auto t1 = std::chrono::steady_clock::now();
            diff += std::chrono::duration<double>(t1 - t0).count();
        }

        double diff_avg = diff / 3.0;
        double gflops = compute_gflops(n, diff_avg);
        std::cout << n << " " << diff_avg << " " << gflops << std::endl;

        n += 24;
    }

    while (n <= 2400) {
        tws::matrix<> A(n, n);
        tws::matrix<> B(n, n);
        tws::matrix<> C(n, n);

        randomize(A);
        randomize(B);
        randomize(C);

        // Warm-up
        func(A, B, C, alpha, beta);

        double diff = 0.0;
        for (int i = 0; i < 3; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            func(A, B, C, alpha, beta);
            auto t1 = std::chrono::steady_clock::now();
            diff += std::chrono::duration<double>(t1 - t0).count();
        }

        double diff_avg = diff / 3.0;
        double gflops = compute_gflops(n, diff_avg);
        std::cout << n << " " << diff_avg << " " << gflops << std::endl;

        n += 120;
    }

    return 0;
}