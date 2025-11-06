#pragma once
#include "twsmatrix.hpp"

double cost_function(
    const tws::vector<double>& predicted, 
    const tws::vector<double>& actual, 
    const tws::matrix<>& control_inputs, 
    const int num_inputs, 
    const double gamma
);

tws::vector<double> finite_difference_gradient(
    const tws::matrix<>& control_inputs,
    const tws::vector<double>& y_target,
    const int num_inputs,
    const double gamma
);

tws::vector<double> backprop_gradient(
    const tws::matrix<>& control_inputs,
    const tws::vector<double>& y_target,
    const tws::vector<double>& y_pred,
    const int num_inputs,
    const int num_states,
    const double gamma
);

void computeGradients();