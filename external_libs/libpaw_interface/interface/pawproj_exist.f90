function pawproj_exist(filename,l_in,ichan_in) result(proj_exist)
!This routine determines if a PAW NL projector exists. Needed for pseudopod.
    use m_pawpsp
    use m_pawtab
    use m_pawrad
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod, only : filename_read, pawsetup=>pawsetup_read, ichan_array=>ichan_array_read, l_array=>l_array_read
    !use libpaw_mod

    implicit none

    character(len=264) :: filename
    integer :: l_in,ichan_in
    integer :: nval
    logical :: proj_exist
    integer :: i,j,l,nchan,l_old
    integer :: st

    proj_exist = .false.

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file to check if paw NL proj exists is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    if(ichan_in < 1) then
        write(*,*) 'ichan_in is less than 1. ichan_in: ', ichan_in
        write(*,*) 'remember that C++ indexes from 0 and Fortran from 1'
        call exit(1)
    endif

    if(l_in < 0) then
        write(*,*) 'l_in is less than 0. l_in: ', l_in
        write(*,*) 'l has to be >= 0'
        call exit(1)
    endif

    nval=pawsetup%valence_states%nval
    do i=1,nval
        if(l_in == l_array(i) .and. ichan_in == ichan_array(i)) proj_exist = .true.
    enddo

end function