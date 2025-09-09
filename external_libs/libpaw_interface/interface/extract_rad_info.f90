integer function pawrad_mesh_size(filename)
!This routine extracts radial grid mesh_size from pawrad object for a specific species
! pawtab_mesh_size is generally different than that of pawrad_mesh_size
    use m_pawpsp
    use m_pawtab
    use m_pawrad
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod, only : pawrad=>pawrad_read, filename_read
    !use libpaw_mod

    implicit none

    character(len=264) :: filename

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file that extract rad mesh size is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    pawrad_mesh_size = pawrad%mesh_size
end function

integer function pawtab_mesh_size(filename)
!This routine extracts radial grid mesh_size from pawtab object for a specific species
! pawtab_mesh_size is generally different than that of pawrad_mesh_size
    use m_pawpsp
    use m_pawtab
    use m_pawrad
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod, only : pawtab=>pawtab_read, filename_read
    !use libpaw_mod

    implicit none

    character(len=264) :: filename

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file that extract rad mesh size is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    pawtab_mesh_size = pawtab%mesh_size
end function


subroutine pawrad_grid(filename,mesh_size,radial_grid)
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
    !use libpaw_mod

    implicit none

    integer, intent(in)    :: mesh_size
    real*8,  intent(inout) :: radial_grid(mesh_size)
    character(len=264) :: filename

    !write(*,*) 'Debug pseudopod'; call flush(6)
    !write(*,*) 'mesh_size ', mesh_size; call flush(6)
    !write(*,*) 'shape(radial_grid )', shape(radial_grid); call flush(6)


    if(filename_read /= filename) then
        write(*,*) 'The pawxml file to extract the radial grid is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    if(mesh_size /= pawrad%mesh_size) then
        write(*,*) 'radial grid mesh sizes not matching extract rad'
        write(*,*) 'input mesh_size: ', mesh_size
        write(*,*) 'pawrad%mesh_size: ', pawrad%mesh_size
        call exit(1)
    endif

    radial_grid = pawrad%rad
    !write(*,*) 'sum(pawrad%rad), pawrad%rad(1:5)', sum(pawrad%rad), pawrad%rad(1:5); call flush(6)
    !write(*,*) 'sum(pawrad%rad), pawrad%rad(1:5)', sum(pawrad%rad(1:pawtab%mesh_size)), pawrad%rad(1:5); call flush(6)
end subroutine