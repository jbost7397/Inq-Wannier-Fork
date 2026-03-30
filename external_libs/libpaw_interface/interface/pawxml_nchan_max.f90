function pawxml_nchan_max(filename) result(nchan_max)
!This routine extracts the largest number of channels with same l for a specific species PAWXML file
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
    integer :: nval
    integer :: nchan_max
    integer :: i,j,l,nchan,l_old
    integer :: st

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file to extract nchan_max is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif
    nval=pawsetup%valence_states%nval
    if(allocated(ichan_array)) deallocate(ichan_array)
    if(allocated(l_array)) deallocate(l_array)
    allocate(l_array(nval),ichan_array(nval),stat=st); if(st/=0) stop 'l_array, ichan_array extract nchan'
    nchan_max = 0
    nchan = 0
    l_array = -1
    ichan_array = -1
    do i=1,nval
        l = pawsetup%valence_states%state(i)%ll
        l_array(i) = l
        nchan = 0
        do j=1,i
            if(l==l_array(j)) then
                nchan=nchan+1
                ichan_array(j) = nchan
            endif
        enddo
        nchan_max = max(nchan_max,nchan)
    enddo

    write(18,*) 'ichan_array:', ichan_array
end function