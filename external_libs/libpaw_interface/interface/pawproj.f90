subroutine pawproj(filename,l_in,ichan_in,mesh_size,proj)
!This routine extracts radial grid info for a specific species
    use m_pawpsp
    use m_pawtab
    use m_pawrad
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod, only : pawrad=>pawrad_read, pawtab=>pawtab_read, filename_read
    use libpaw_mod, only : ichan_array=>ichan_array_read, l_array=>l_array_read
    !use libpaw_mod

    implicit none

    integer, intent(in)    :: mesh_size, l_in, ichan_in
    real*8,  intent(inout) :: proj(mesh_size)
    character(len=264) :: filename
    integer :: nval,i
    logical, external :: pawproj_exist

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file to extract the radial grid is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    if(mesh_size /= pawtab%mesh_size) then
        write(*,*) 'radial grid mesh sizes not matching extract_tproj'
        write(*,*) 'input mesh_size: ', mesh_size
        write(*,*) 'pawtab%mesh_size: ', pawtab%mesh_size
        call exit(1)
    endif

    if(.not. pawproj_exist(filename,l_in,ichan_in)) then
        write(*,*) "PAW NL projector for l, ichan ", l_in, ichan_in, " doesn't exist"
        call exit(1)
    endif 

    nval = pawtab%basis_size
    do i=1,nval
        if(l_in == l_array(i) .and. ichan_in == ichan_array(i)) then
            proj(:) = pawtab%tproj(:,i)
        endif 
    enddo
end subroutine