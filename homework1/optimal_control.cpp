#include <iostream>
#include "twsmatrix.hpp"
#include "optimal_control.hpp"
#include "simulate_system.hpp"
#include "compute_gradient.hpp"

void gradient_descent(tws::matrix<>& u,
                      int num_inputs,
                      int num_states,
                      double gamma,
                      double alfa,
                      double eps
                    )
{
    double J_old = std::numeric_limits<double>::infinity();
    tws::vector<double> y(num_inputs);
    int count = 0;
    
    while (count < 100000) {
        tws::vector<double> x = initial_state;
        count++;
        // 1) Forward pass: y(u)
        auto y_new = computeOutput(num_inputs, u, num_states, A, B, C, x);
        y = y_new;


        // 2) Cost
        double J_new = cost_function(y, y_target, u, num_inputs, gamma);

        // 3) Backprop gradient dJ/du (length N)
        tws::vector<double> g = backprop_gradient(u, y_target, y, num_inputs, num_states, gamma);
        //tws::vector<double> g = finite_difference_gradient(u, y_target, num_inputs, gamma);

        // 4) Gradient step on u (u_k ← u_k − α * g_k)
        for (int k = 0; k < num_inputs; ++k) {
            u(0, k) -= alfa * g[k];
        }
        
        if (std::abs(J_new - J_old) < eps){
            //std::cout << "Converged after " << count << " iterations.\n";
            break;
        }
            

        J_old = J_new;
    }

    print_matrix(u);
    print_vector(y);

}
