1. The matrix A that the matvec function represents is the resulting matrix from the calculation y = Ax. The matrix A is the diagonal, with all off-entries are 0.


3. Using matvec and matvec1 for float:
orig error: 5.74199e-11
orig error: 2.04745e-07
relative error: 5.74199e-11
relative error: 2.04745e-07

I do get the same results regardless of using the struct/functor or the regular/using the 

4. By using the functor the cg typename Op will then take the function as either vector<double> or vector<float> depending on the function passed.

6. We can not implement a function matvec4(x, y, m), due to how Op is defined in cg.hpp: op(p, q), meaning the tird prameter m will not be utilized.