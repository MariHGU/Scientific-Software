#pragma once
#include "twsmatrix.hpp"

void gradient_descent(tws::matrix<>& u,
                      int num_inputs,
                      int num_states,
                      double gamma,
                      double alfa,       
                      double eps
                    );


