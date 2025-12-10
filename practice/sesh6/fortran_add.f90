!> Adds two arrays element-wise.
!!
!! @param[in]    n The number of elements in the arrays.
!! @param[in]    a The first array.
!! @param[inout] b The second array.
subroutine fortran_add(n, a, b)
    implicit none
    integer, intent(in) :: n
    double precision, intent(in) :: a(*)
    double precision, intent(inout) :: b(*)

    integer :: i

    do i = 1, n
        b(i) = a(i) + b(i)
    end do

end subroutine fortran_add 