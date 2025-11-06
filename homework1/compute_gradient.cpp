#include <iostream>
#include "twsmatrix.hpp"
#include "compute_gradient.hpp"
#include "simulate_system.hpp"


double cost_function(const tws::vector<double>& predicted, const tws::vector<double>& actual, const tws::matrix<>& control_inputs, const int num_inputs, const double gamma) {
    /**
     * @brief Compute the cost as the sum of squared differences between predicted and actual matrices.
     * 
     * @param predicted The predicted output matrix.
     * @param actual The actual output matrix.
     * @return double The computed cost.
     **/
    double cost = 0.0;
    double diff = 0.0;
    double input_penalty = 0.0;

    for (int k = 0; k < num_inputs; ++k) {
        diff += (predicted[k] - actual[k]) * (predicted[k] - actual[k]);
        input_penalty += control_inputs(0, k) * control_inputs(0, k);
    }
    cost = (diff + gamma * input_penalty)/num_inputs;

    return cost;
}


tws::vector<double> finite_difference_gradient(
    const tws::matrix<>& control_inputs,
    const tws::vector<double>& y_target,
    const int num_inputs,
    const double gamma
) {
    double h = 1e-6;
    double factor = 1.0 / (2.0 * h);
    tws::vector<double> finDiff_grad(num_inputs);

    for (int k = 0; k < num_inputs; ++k) {
        tws::matrix<> u = control_inputs;
        double original_value = u(0, k);

        u(0, k) = original_value + h;
        auto y_plus = computeOutput(
            num_inputs, u, num_states, A, B, C, initial_state
        );
        double cost_plus_h = cost_function(y_plus, y_target, u, num_inputs, gamma);

        u(0, k) = original_value - h;
        auto y_min = computeOutput(
            num_inputs, u, num_states, A, B, C, initial_state
        );
        double cost_minus_h = cost_function(y_min, y_target, u, num_inputs, gamma);

        u(0, k) = original_value;
        

        finDiff_grad[k] = (cost_plus_h - cost_minus_h) * factor;
    }

    return finDiff_grad;
}

tws::vector<double> backprop_gradient(
    const tws::matrix<>& control_inputs,
    const tws::vector<double>& y_target,
    const tws::vector<double>& y_pred,
    const int num_inputs,
    const int num_states,
    const double gamma
) {
    tws::vector<double> grad(num_inputs);    
    tws::matrix<> lambda(num_states, num_inputs + 1);

    for (int i = 0; i < num_states; ++i) {
        lambda(i, num_inputs) = 0.0;  // Terminal condition
    }
    
    auto AT = tws::transpose(A);
    
    const double f = 2.0 / num_inputs;
    
    tws::vector<double> lambda_kp1(num_states);
    tws::vector<double> ATv(num_states);
    tws::vector<double> lambda_k(num_states);

    for (int k = num_inputs - 1; k >= 0; --k) {
        // CT contribution: C(0,i) * f * (y_pred[k] - y_target[k])
        double dcost_dy_scalar = f * (y_pred[k] - y_target[k]);
        for (int i = 0; i < num_states; ++i) {
            lambda_k[i] = C(0, i) * dcost_dy_scalar; // CT_contribution
        }

        // Load lambda[k+1]
        for (int i = 0; i < num_states; ++i) {
            lambda_kp1[i] = lambda(i, k + 1);
        }

        // Compute A^T * lambda[k+1] without transposing A
        for (int i = 0; i < num_states; ++i) {
            ATv[i] = 0.0;
            for (int j = 0; j < num_states; ++j) {
                ATv[i] += A(j, i) * lambda_kp1[j];
            }
        }
        // lambda[k] = CT_contribution + A^T * lambda[k+1]
        for (int i = 0; i < num_states; ++i) {
            lambda(i, k) = lambda_k[i] + ATv[i];
        }
        // Compute dcost/du[k] = (2gamma/N) * u[k] + BT * lambda[k+1]
        grad[k] = f * gamma * control_inputs(0, k) + tws::dot(B, lambda_kp1);
        
    }
    
    return grad;
}

void computeGradients(){
    tws::vector<double> y(num_inputs);
    auto y_new = computeOutput(num_inputs, control_inputs, num_states, A, B, C, initial_state);
    y = std::move(y_new);


    tws::vector<double> finGrad = finite_difference_gradient(control_inputs, y_target, num_inputs, gamma);
    for (int i = 0; i < num_inputs; ++i) std::cout << finGrad[i] << " ";
    std::cout << std::endl;

    tws::vector<double> backPropGrad = backprop_gradient(control_inputs, y_target, y, num_inputs, num_states, gamma);
    for (int i = 0; i < num_inputs; ++i) std::cout << backPropGrad[i] << " ";
    std::cout << std::endl;

}