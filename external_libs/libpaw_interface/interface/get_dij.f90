subroutine calculate_dij(natom,ntypat,ixc,xclevel,nfft,nspden,xred,ucvol,gprimd,vtrial,vxc,epawdc)
    use libpaw_mod
    use m_paw_denpot
    use m_pawdij

    implicit none

    real*8  :: compch_sph, epaw, epawdc
    real*8  :: nucdipmom(3,natom), qphon(3)
    real*8  :: vtrial(cplex*nfft,nspden),vxc(cplex*nfft,nspden)

    integer :: natom,ntypat,ixc,xclevel,iatom
    integer :: nfft, nspden
    real*8  :: xred(3,natom),ucvol,gprimd(3,3)

    write(*,*) '4. Generating dij from on-site rhoij, v_ks and v_xc'

    ! Note : in Abinit PAW calculations, v_h is calculated with ncomp + ntilde + ncoretilde
    ! while v_xc is calculated with ntilde + ncoretilde by default

    nucdipmom = 0.0
    qphon = 0.0

    call pawdenpot(compch_sph, epaw, epawdc, 0, ixc, natom, natom, & !ipert
        & nspden, ntypat, nucdipmom, 0, 0, paw_an, paw_an, paw_ij, & !nzlmopt, option
        & pawang, 0, pawrad, pawrhoij, 0, pawtab, xcdev, 1d0, xclevel, & !pawprtvol, pawspnorb, spnorbscl
        & denpos, ucvol, znucl)
    
    !write(21,*) 'epaw',epaw
    !write(21,*) 'epawdc',epawdc

    do iatom = 1,natom
        paw_ij(iatom)%has_dij=1 !reset the flag so dij will be calculated again
    enddo
    call pawdij(cplex, 0, gprimd, 0, natom, natom, nfft, nfft, nspden, ntypat, & !enunit, ipert
        & paw_an, paw_ij, pawang, pawfgrtab, 0, pawrad, pawrhoij, 0, pawtab, & !pawprtvol, pawspnorb
        & xcdev, qphon, 1d0, ucvol, 0d0, vtrial, vxc, xred) !spnorbscl, charge

end subroutine

subroutine get_dij(iatom,size_dij,nspden,dij)
    use libpaw_mod
    implicit none
    integer :: iatom, size_dij, nspden
    real*8  :: dij(size_dij,nspden)

    dij = paw_ij(iatom)%dij
end subroutine

subroutine get_sij(itype,size_dij,sij)
    use libpaw_mod
    implicit none
    integer :: itype,size_dij
    real*8  :: sij(size_dij)

    sij = pawtab(itype)%sij
end subroutine