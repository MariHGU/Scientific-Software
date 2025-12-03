#ifndef tws_cg_hpp
#define tws_cg_hpp

#include <cassert>
#include <type_traits>

#include "twsmatrix.hpp"

namespace tws {

/**
 * This is code for the conjugate gradient Krylov method for solving a symmetric
 * linear system (that is guaranteed to work for positive matrices)
 *
 * Stopcriterion:
 *    number of iterations <= maximum number of iterations max_it
 *    ||b - Ax||_2 <= tolerance * ||b||_2
 *
 * @param[in]    op functor
 *                  op( x, y ) computes y = A * x, i.e., op is a linear operator
 *                  where x and y are Vectors.
 *
 * @param[inout] x  Vector
 *                  On input: initial guess
 *                  On output: solution to the linear system
 *
 * @param[inout] b  Vector
 *                  On input: right-hand side
 *                  On output: residual of the final solution: r = b - A * x
 *
 * @param tolerance floating point number
 *                  residual tolerance of the linear solver
 *                  default value: 1.0e-5
 *
 * @param max_it    integer
 *                  maximum number of iterations
 *                  default value: 100
 *
 * @return          floating point number
 *                  residual of the final solution: ||b - Ax||_2
 */
template <typename T, typename Op>
double cg(const Op& op,
          vector<T>& x,
          vector<T>& b,
          double tolerance = 1.0e-5,
          int max_it = 100)
{
    assert(x.size() == b.size());
    assert(tolerance > 0.);
    assert(max_it > 0);

    vector<T> p(x.size());
    vector<T> q(x.size());

    T norm_b_2 = dot(b, b);

    op(x, q);
    b -= q;

    T norm_r_2 = dot(b, b);
    if (norm_r_2 < tolerance * tolerance * norm_b_2) return norm_r_2;

    p = b;
    for (int iter = 0; iter < max_it; ++iter) {
        op(p, q);
        T alpha = norm_r_2 / dot(p, q);

        x += alpha * p;
        b -= alpha * q;

        T norm_r_2_new = dot(b, b);
        if (norm_r_2_new < tolerance * tolerance * norm_b_2)
            return norm_r_2_new;

        T beta = norm_r_2_new / norm_r_2;
        p = b + beta * p;
        norm_r_2 = norm_r_2_new;
    }

    return norm_r_2;
}
}  // namespace tws

#endif
