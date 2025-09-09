function pawxml_zval(filename) result(zval)
!This routine extracts radial grid info for a specific species
    use m_pawpsp
    use m_pawtab
    use m_pawrad
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod, only : zion_read, filename_read
    !use libpaw_mod

    implicit none

    character(len=264) :: filename
    real*8  :: zval

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file that extract zval/valence charge is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    zval = zion_read
end function