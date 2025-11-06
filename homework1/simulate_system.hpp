#pragma once
#include <vector>
#include <string>
#include "twsmatrix.hpp"

extern tws::matrix<double> A;
extern tws::vector<> B;
extern tws::matrix<> C;

extern int num_states;
extern int num_inputs;
extern double gamma;
extern double alfa;
extern double eps;

extern tws::vector<> initial_state;
extern tws::matrix<> control_inputs;

extern tws::vector<double> y_target;

tws::vector<double> computeOutput(
    int num_inputs,
    const tws::matrix<>& control_inputs,
    int num_states,
    const tws::matrix<double>& A,
    const tws::vector<>& B,
    const tws::matrix<>& C,
    const tws::vector<>& initial_state
);

void initSystem();
