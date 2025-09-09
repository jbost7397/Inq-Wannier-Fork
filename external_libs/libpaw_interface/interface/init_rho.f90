subroutine init_rho(nspden,ngfftdg,nfft,natom,ntypat,rprimd,gprimd,gmet,ucvol,xred,rho)
    use libpaw_mod
    use m_kg
    use m_mkrho

    implicit none

    integer :: nspden
    real*8  :: rho(nfft,nspden)
    real*8, allocatable :: ph1d(:,:)
    real*8  :: rcut
    integer :: nfft, mgrid, vprtrb(2), qprtrb(3)
    integer :: ngfftdg(3)
    integer :: natom, ntypat
    real*8  :: rprimd(3,3),gprimd(3,3),gmet(3,3)
    real*8  :: ucvol,xred(3,natom)
    real*8  :: densty(ntypat,4), spinat_in(3,natom)

    mgrid = maxval(ngfftdg)

    allocate(ph1d(2,3*(2*mgrid+1)*natom))
    ph1d = 0d0
    qprtrb = 0
    rcut = 0d0
    vprtrb = 0d0
    densty = 0d0
    spinat_in = 0d0 ! to be set later!!

    call getph(atindx,natom,ngfftdg(1),ngfftdg(2),ngfftdg(3),ph1d,xred)

    call initro(pawtab,atindx,densty,gmet,gsqcutdg,1,mgrid,mqgrid,natom,nattyp,&
        nfft,ngfftdg,nspden,ntypat,ph1d,qgrid_vl,rho,spinat_in,ucvol,1,zion,znucl,&
        ngfftdg(2),fftn2_distrib,ffti2_local,ngfftdg(3),fftn3_distrib,ffti3_local)
end subroutine