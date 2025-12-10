void fortran_matvec_(const int* m, const int* n, const double* A, const int* lda, const double* x, double* y){
    for (int i=0; i<*m; ++i){
        double sum = 0.0;
        for (int j=0; j<*n; ++j){
            sum += A[i + j*(*lda)]*x[j];
        }
        y[i] = 2.0*sum;
    }
}