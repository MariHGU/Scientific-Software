#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>

#include "twsmatrix.hpp"
#include "simulate_system.hpp"
#include "compute_gradient.hpp"
#include "optimal_control.hpp"


// A: 2x2 matrix, B = 2x1, C = 1x2, inital sizes, but will be re-initialized to work for any size
tws::matrix<double> A(2, 2);
tws::vector<> B(2);
tws::matrix<> C(1, 2);

int num_states;
int num_inputs;
double gamma;
double alfa;
double eps;

tws::vector<> initial_state(2);
tws::matrix<> control_inputs(1, 5);
tws::vector<double> y_target(5);

std::string filepath = "inputs.txt";


void stripLine(std::string& line){
    auto pos = line.find('#');
    if (pos != std::string::npos) {
        line.erase(pos);
    }
}

std::vector<std::string> readInputs(const std::string filepath) {
    std::vector<std::string> lines;
    lines.reserve(20);
    std::string line;

    std::ifstream infile(filepath);

    while (std::getline(infile, line)) {
        stripLine(line);
        lines.push_back(line);
    }
    return lines;
}

void initMatricies(std::vector<std::string> inputs){
    /**
     * @brief matricies based on input strings.
     **/
    for (int i=0; i<inputs.size(); i++){
        //std::cout << inputs[i] << std::endl;
        switch (i){
            case 0:
                num_states = std::stoi(inputs[i]);
                A = tws::matrix<double>(num_states, num_states);
                B = tws::vector<>(num_states);
                C = tws::matrix<>(1, num_states);

                initial_state = tws::vector<>(num_states);
                break;
            case 1:
                num_inputs = std::stoi(inputs[i]);
                control_inputs = tws::matrix<>(1, num_inputs);
                y_target = tws::vector<>(num_inputs);
                break;
            case 2:
                // Matrix A
                {
                    std::istringstream iss(inputs[i]);
                    for (int c = 0; c < num_states; ++c) {
                        for (int r = 0; r < num_states; ++r) {
                            iss >> A(r, c);
                        }
                    }
                    //print_matrix(A);
                }
                break;
            case 3:
                // Matrix B
                {
                    std::istringstream iss(inputs[i]);
                    for (int r = 0; r < num_states; ++r) {
                        iss >> B[r];
                    }

                    //print_vector(B);
                }
                break;
            case 4:
                // Matrix C
                {
                    std::istringstream iss(inputs[i]);
                    for (int c = 0; c < num_states; ++c) {
                        iss >> C(0, c);
                    }
                    //print_matrix(C);
                }
                break;
            case 5:
                // Initial state
                {
                    std::istringstream iss(inputs[i]);
                    for (int r = 0; r < num_states; ++r) {
                        iss >> initial_state[r];
                    }
                    //print_vector(initial_state);
                }
                break;
            
            case 6:
                // Control inputs
                {
                    std::istringstream iss(inputs[i]);
                    for (int j = 0; j < num_inputs; ++j) {
                        iss >> control_inputs(0, j);
                    }
                    //print_matrix(control_inputs);
                }
                break;
            case 7:
                // Target y
                {
                    std::istringstream iss(inputs[i]);
                    for (int j = 0; j < num_inputs; ++j) {
                        iss >> y_target[j];
                    }
                    //print_vector(y_target);
                }
                break;
            case 8:
                // Gamma
                {
                    std::istringstream iss(inputs[i]);
                    iss >> gamma;
                }
                break;
            case 9:
                // Alfa
                {
                    std::istringstream iss(inputs[i]);
                    iss >> alfa;
                }
                break;
            case 10:
                // Eps
                {
                    std::istringstream iss(inputs[i]);
                    iss >> eps;
                }
                break;
            default:
                break;
        }
    }
}

tws::vector<double> computeOutput(
    int num_inputs,
    const tws::matrix<>& control_inputs,
    int num_states,
    const tws::matrix<double>& A,
    const tws::vector<>& B,
    const tws::matrix<>& C,
    const tws::vector<>& initial_state
){
    /**
     * @brief the output-vector y for the LTI system over a series of control inputs u_k.
     **/
    tws::vector<double> outputs(num_inputs);
    tws::vector<double> x = initial_state;

    tws::vector<double> Ax(num_states);
    tws::vector<double> Bu(num_states);
    tws::vector<double> y_vec(1); // C is 1×n, so y is scalar

    for (int i = 0; i < num_inputs; ++i) {
        const double u = control_inputs(0, i);   // scalar u_k

        tws::multiply(A, x, Ax);
        tws::multiply(B, u, Bu);
        
        y_vec = C * x;
        x = Ax + Bu;
        outputs[i] = y_vec[0];
    }
    return outputs;
}

void initSystem(){
    std::vector<std::string> inputs = readInputs(filepath);
    initMatricies(inputs);

    tws::vector<double> y(num_inputs);
    auto y_new = computeOutput(num_inputs, control_inputs, num_states, A, B, C, initial_state);
    y = std::move(y_new);

    for (int i = 0; i < num_inputs; ++i) std::cout << y[i] << " ";
    std::cout << std::endl;

}

int main(){
    initSystem();

    computeGradients();

    gradient_descent(control_inputs, num_inputs, num_states, gamma, alfa, eps);

    return 0;
}
