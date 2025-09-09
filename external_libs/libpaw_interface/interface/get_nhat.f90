subroutine get_nhat(natom,ntypat,xred,ngfft,nfft,nspden,gprimd,rprimd,ucvol,nhat,nhatgr)
    use libpaw_mod
    use m_paw_nhat
    implicit none

    integer :: iatom,nfft,nspden
    real*8  :: compch_fft
    real*8  :: nhat(nfft,nspden), nhatgr(nfft,nspden,3)
    real*8  :: qphon(3)
    real*8  :: gprimd(3,3),rprimd(3,3)
    integer :: natom,ntypat,ngfft(3)
    real*8  :: ucvol,xred(3,natom)

    ! Some matrix elements of the on-site density matrix rhoij
    ! might be zero, and only the non-zero elements are stored in
    ! pawrhoij%rhoijp. The number of non-zero elements is given by
    ! pawrhoij%nrhoijsel, and the indexes of those elements are stored in
    ! pawrhoij%rhoijselect

    ! But because these arrays are not reset during SCF run,
    ! this cannot be directly seen from the current values of rhoijp

    write(*,*) '3. Generating compensation charge from on-site rhoij'
    
    qphon = 0.0 !phonon, not used

    call pawmknhat(compch_fft, cplex, 0, 0, 0, 0, & !ider,idir,ipert,izero
        & gprimd, natom, natom, nfft, ngfft, 0, &!nhatgrdim
        & nspden, ntypat, pawang, pawfgrtab, nhatgr, nhat, &
        & pawrhoij, pawrhoij, pawtab, qphon, rprimd, ucvol, 0, xred) !usewvl

    !write(19,*) nhat
end subroutine