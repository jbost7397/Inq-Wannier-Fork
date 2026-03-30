subroutine set_rhoij(iatom,nrhoijsel,size_rhoij,nspden,rhoijselect,rhoijp)
    use libpaw_mod
    implicit none

    integer :: nrhoijsel,iatom,size_rhoij,nspden
    integer :: rhoijselect(size_rhoij)
    real*8  :: rhoijp(size_rhoij,nspden)

    pawrhoij(iatom)%nrhoijsel = nrhoijsel
    pawrhoij(iatom)%rhoijselect = rhoijselect
    pawrhoij(iatom)%rhoijp = rhoijp

    write(*,*) 'On-site Density Matrix for atom ',iatom
    call pawrhoij_print_rhoij(pawrhoij(iatom)%rhoijp,pawrhoij(iatom)%cplex_rhoij,&
    &                  pawrhoij(iatom)%qphase,iatom,size(pawrhoij),&
    &                  rhoijselect=pawrhoij(iatom)%rhoijselect,unit=6,&
    &                  opt_prtvol=-1,mode_paral='COLL')


end subroutine