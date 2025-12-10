!> Fortran subroutine that computes the matrix-vector product y = A*x.
!!
!! @param[in] m   integer
!!                Number of rows of matrix A.
!! @param[in] n   integer
!!                Number of columns of matrix A.
!! @param[in] A   m x n matrix
!!                Matrix A.
!! @param[in] lda integer
!!                Leading dimension of A.
!! @param[in] x   size n vector
!!                Vector x.
!! @param[out] y  size m vector
!!                Vector y.
subroutine fortran_matvec(m, n, A, lda, x, y)
    implicit none

    ! Arguments
    integer, intent(in) :: m, n, lda
    double precision, intent(in) :: A(lda, *), x(*)
    double precision, intent(out) :: y(*)

    ! Local variables
    integer :: i, j

    ! Compute y = A*x
    do i = 1, m
        y(i) = 0.0
        do j = 1, n
            y(i) = y(i) + A(i, j) * x(j)
        end do
    end do

end subroutine fortran_matvec