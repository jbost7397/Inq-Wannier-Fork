function pawxml_lmax(filename) result(lmax)
!This routine extracts the largest angular momentum quantum number l for a specific species PAWXML file
    use m_pawpsp
    use m_pawtab
    use m_pawrad
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod, only : lmax_read, filename_read
    !use libpaw_mod

    implicit none

    character(len=264) :: filename
    integer :: lmax

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file that extract lmax is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    lmax = lmax_read
end function