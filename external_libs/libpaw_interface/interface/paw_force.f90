! This subroutine calls pawgrnl, which gives term F2 (Eq. 31)
! of the PAW paper (Torrent 2008)

subroutine paw_force(natom,ntypat,typat,ngfftdg,nfft,nhat,xred,rprimd,gprimd,gmet,ucvol,vtrial,vxc,rhor,nspden, &
    grad,stress)
    use libpaw_mod
    use m_kg
    use m_paw_dfpt
    use m_sg2002
    use m_atm2fft
    implicit none

    integer :: ngfftdg(3)
    integer :: nfft,nspden,natom

    real*8  :: nhat(nfft,nspden),vtrial(nfft,nspden),vxc(nfft,nspden)

    real*8  :: grad(3*natom),stress(6)
    real*8  :: grnl(3*natom),nlstr(3*natom) ! eqn. 31 of torrent08
    real*8  :: grl(3*natom) ! eqn. 30 of torrent08
    real*8  :: grxc(3*natom) ! eqn. 33 of torrent08
    integer :: mgrid, ntypat, typat(natom)
    real*8, allocatable :: ph1d(:,:)
    real*8, allocatable :: rhog(:,:),vxc_tot(:),vxcg(:,:)
    real*8  :: rhor(nfft)

    real*8  :: qphon(3),rcut
    real*8  :: rprimd(3,3),gprimd(3,3),xred(3,natom),gmet(3,3),ucvol
    integer :: qprtrb(3)

    real*8, allocatable :: dummy(:),dummy1(:),dummy2(:),dummy3(:),dummy4(:),dummy5(:),dummy6(:),dummy7(:),dummy8(:)

    mgrid = maxval(ngfftdg)
    allocate(ph1d(2,3*(2*mgrid+1)*natom))
    ph1d = 0d0
    qphon = 0.0
    qprtrb = 0.0
    rcut = 0.0

    ! transform rho and vxc to g space
    allocate(vxc_tot(nfft))
    if(nspden == 1) vxc_tot = vxc(:,1)
    if(nspden == 2) vxc_tot = 0.5d0*(vxc(:,1)+vxc(:,2))

    allocate(rhog(2,nfft),vxcg(2,nfft))
    call sg2002_mpifourdp(1,nfft,ngfftdg,1,-1,&
        fftn2_distrib,ffti2_local,fftn3_distrib,ffti3_local,rhog,rhor,0)
    call sg2002_mpifourdp(1,nfft,ngfftdg,1,-1,&
        fftn2_distrib,ffti2_local,fftn3_distrib,ffti3_local,vxcg,vxc_tot,0)

    call getph(atindx,natom,ngfftdg(1),ngfftdg(2),ngfftdg(3),ph1d,xred)

    grnl = 0.0
    nlstr = 0.0
    call pawgrnl(atindx1,nspden,grnl,gsqcut,mgrid,natom,natom,nattyp,nfft,ngfftdg,nhat, &
        nlstr,nspden,ntypat,1,0,1,0,pawang,pawfgrtab,pawrhoij,pawtab,ph1d, &
        qphon,rprimd,gmet,typat,ucvol,vtrial,vxc,xred)

    grl = 0.0
    grxc = 0.0
    call atm2fft(atindx1,dummy,dummy1,dummy2,dummy3,dummy4,dummy5,gmet,gprimd,grxc,grl, &
        gsqcutdg,mgrid,mqgrid,natom,nattyp,nfft,ngfftdg,ntypat,0,0,0,1,1,1,0,1, &
        pawtab,ph1d,qgrid_vl,qprtrb,rcut,rhog,rprimd,dummy6,dummy7,ucvol,1, &
        vxcg,vxcg,vxcg,dummy8,vlspl,&
        ngfftdg(2),fftn2_distrib,ffti2_local,ngfftdg(3),fftn3_distrib,ffti3_local)

    deallocate(rhog)
    deallocate(vxc_tot)
    deallocate(vxcg)

    grad = grl + grxc + grnl

end subroutine