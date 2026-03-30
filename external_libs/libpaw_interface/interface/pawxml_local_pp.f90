subroutine pawxml_local_pp(filename,mesh_size,loc_pot)
!This routine extracts the Kresse-Joubert local ionic pseudopotential 
! i.e. the Hartree potential for pseudized Zc density, v_H[\tilde{n}_{Zc}].
! Libpaw performs internal conversion to calculate the above from the 
! Blochl local ionic pseudopotential if that is what is provided in the PAWXML file.
! Refer to Eq.(22) of Torrent, Marc, et al. "Implementation of the projector augmented-wave method 
! in the ABINIT code: Application to the study of iron under pressure." 
! Computational Materials Science 42.2 (2008): 337-351.
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

    integer, intent(in)    :: mesh_size
    real*8,  intent(inout) :: loc_pot(mesh_size)
    character(len=264) :: filename

    if(filename_read /= filename) then
        write(*,*) 'The pawxml file to extract the radial grid is supposed to be reading: ', filename_read
        write(*,*) 'but received the following instead: ', filename
        call exit(1)
    endif

    if(mesh_size /= size(pawtab%vhtnzc)) then
        write(*,*) 'radial grid mesh sizes not matching extract_local_pseudo'
        write(*,*) 'mesh_size has to be equal to pawtab%mesh_size not pawrad%mesh_size'
        write(*,*) 'input mesh_size: ', mesh_size
        write(*,*) 'size(pawtab%vhtnzc): ', size(pawtab%vhtnzc)
        write(*,*) 'pawtab%mesh_size: ', pawtab%mesh_size
        call exit(1)
    endif

    loc_pot = pawtab%vhtnzc
    write(*,*) 'sum(locpot)', sum(loc_pot); call flush(6)
    write(*,*) 'locpot(1:5)', loc_pot(1:5); call flush(6)
end subroutine