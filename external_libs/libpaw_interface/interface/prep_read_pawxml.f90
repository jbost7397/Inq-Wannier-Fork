subroutine prep_read_pawxml(filename)
    !Calling a bunch of Libpaw routines to extract the projectors and their radial grids.
    !The projectors (tproj) are different than the projectors extracted from the 
    !paw_xml files because Abinit performs additional splining and defines its own
    !radial grids around each species.
    use m_pawpsp
    use m_pawtab
    use m_pawrad
    use m_pawxmlps
    use m_paw_init
    use m_kg
    use m_paw_occupancies
    use m_paw_nhat
    use libpaw_mod, only : pawrad=>pawrad_read, pawtab=>pawtab_read, pawsetup=>pawsetup_read
    use libpaw_mod, only : lmax=>lmax_read, filename_read, zion=>zion_read
    !use libpaw_mod

    implicit none

    integer :: stat
    integer :: n3, i3, n2, i2
    integer :: it, ia
    integer :: iproj,imax

    real*8  :: ecut, ecutpaw !a coarse grid, and a fine grid for PAW
    real*8  :: gmet(3,3) !reciprocal space lattice vectors
    real*8  :: rprimd(3,3) !lattice vectors
    real*8  :: gprimd(3,3) !reciprocal space lattice vectors
    integer :: ngfft(3),ngfftdg(3)
    real*8  :: ucvol !volume of unit cell
    integer :: ixc, xclevel !functional; will be set externally in practice
    real*8  :: epsatm
    integer :: nspden,nsppol!number of spin components for density and wavefunctions
    !for normal nspin = 1,2 calculations, nsppol is set to be same as nspden

    character(len=264) :: filename

    !libpaw_mod internal versions to isolate routines
    integer :: lloc, pspcod, pspxc
    integer :: nattyp
    integer :: lexexch, lpawu !no exact exchange, no dft+u
    integer, allocatable :: atindx1(:),atindx(:)
    !type(pawtab_type)   :: pawtab !set to 1 for now, will expand later
    !type(pawrad_type)   :: pawrad
    type(pawpsp_header_type) :: pawpsp_header
    !type(paw_setup_t)        :: pawsetup
    real*8,allocatable  :: qgrid_ff(:), qgrid_vl(:) !ff:'corase', vl:'fine'
    real*8,allocatable  :: ffspl(:,:,:), vlspl(:,:)
    real*8  :: znucl
    real*8  :: gsqcut, gsqcutdg, gsqcut_eff
    real*8  :: hyb_mixing, hyb_range_fock
    real*8  :: r2well
    real*8  :: xcccrc

    ! Some default values in ABINIT
    integer :: iboxcut = 0
    integer :: mqgrid = 3001, lnmax = 6, ipsp = 1
    integer :: usexcnhat = 0, xcdev = 1, usewvl = 0, icoulomb = 0, usepotzero = 0
    real*8  :: denpos = 1d-14
    integer :: gnt_option = 1, lcutdens = 10, lmix = 10
    integer :: mpsang, nphi = 13, ntheta = 12, nsym = 1 !the spherical grid
    real*8  :: effmass_free = 1.0
    integer :: cplex = 1

    filename_read = filename
    hyb_mixing = 0.0
    hyb_range_fock = 0.0

    !Arbitrary choices that don't affect the values of the certain pawtab object we want to extract in pseudopod
    ecut     = 10; ecutpaw=10
    gmet     = reshape((/1d-2,0d0,0d0,0d0,1d-2,0d0,0d0,0d0,1d-2/), (/3,3/))
    rprimd   = reshape((/1d2,0d0,0d0,0d0,1d2,0d0,0d0,0d0,1d2/)   , (/3,3/))
    gprimd   = reshape((/1d-1,0d0,0d0,0d0,1d-1,0d0,0d0,0d0,1d-1/), (/3,3/))
    ucvol    = 1000
    ngfft    = (/30,30,30/)
    ngfftdg  = (/30,30,30/)
    ixc      = 7
    xclevel  = 1
    nspden   = 1
    nsppol   = 1


    ! not supporting dftu for now!
    lexexch = 0
    lpawu = 0

    ! Process energy cutoff
    call getcut(ecut,gmet,gsqcut,iboxcut,ngfft)
    call getcut(ecutpaw,gmet,gsqcutdg,iboxcut,ngfftdg)
    
    allocate(qgrid_ff(mqgrid),qgrid_vl(mqgrid),stat=stat)
    if(stat/=0) then
        write(*,*) 'problem allocating mqgrid'
        call exit(1)
    endif
    
    call generate_qgrid(gsqcut,qgrid_ff,mqgrid)
    call generate_qgrid(gsqcutdg,qgrid_vl,mqgrid)

    allocate(ffspl(mqgrid,2,lnmax), vlspl(mqgrid,2))

    ! Read paw input files
    call paw_setup_free(pawsetup)
    call paw_setup_free(paw_setuploc)
    call rdpawpsxml(filename, pawsetup)
    call rdpawpsxml(filename, paw_setuploc)
    call pawpsp_read_header_xml(lloc, lmax, pspcod, pspxc,&
        & pawsetup, r2well, zion, znucl)
    call pawpsp_read_pawheader(pawpsp_header%basis_size,&
        &   lmax,pawpsp_header%lmn_size,&
        &   pawpsp_header%l_size, pawpsp_header%mesh_size,&
        &   pawpsp_header%pawver, pawsetup,&
        &   pawpsp_header%rpaw, pawpsp_header%rshp, pawpsp_header%shape_type)

    ! Process onsite information
    call pawtab_nullify(pawtab)
    call pawtab_set_flags(pawtab,has_tvale=1,has_vhnzc=1,has_vhtnzc=1,has_tproj=1)
    call pawpsp_17in(epsatm, ffspl, icoulomb, ipsp, hyb_mixing, ixc, lmax,&
            &       lnmax, pawpsp_header%mesh_size, mqgrid, mqgrid, pawpsp_header,&
            &       pawrad, pawtab, xcdev, qgrid_ff, qgrid_vl, usewvl, usexcnhat,&
            &       vlspl(:,:), xcccrc, xclevel, denpos, zion, znucl)
    !call paw_setup_free(pawsetup)
    !call paw_setup_free(paw_setuploc)
    
    !zion_read = zion

    imax=pawrad_ifromr(pawrad,pawtab%rpaw)
    !write(15,*) 'pawrad%mesh_type, pawrad%mesh_size',pawrad%mesh_type, pawrad%mesh_size
    write(15,*) 'shape(pawrad%rad), pawrad%mesh_size',shape(pawrad%rad), pawrad%mesh_size
    write(15,*) 'sum(pawrad%rad)',sum(pawrad%rad)
    
    write(15,*) 'it, imax, basis_size, shape(tprojs): ',it, imax, pawtab%basis_size, shape(pawtab%tproj)
    call flush(15)
    write(15,*) 'iproj, sum(pawtab%tproj(:,iproj))'; call flush(13)
    do iproj = 1,pawtab%basis_size
        write(15,*) iproj, sum(pawtab%tproj(:imax,iproj))
    enddo

    write(17,*) 'nval', pawsetup%valence_states%nval; call flush(17)
    write(17,*) 'iproj, pawsetup%valence_state%ll'; call flush(17)
    write(17,*) 1, shape(pawsetup%valence_states%state); call flush(17)
    do iproj=1,pawsetup%valence_states%nval
        write(17,*) iproj, pawsetup%valence_states%state(iproj)%ll
    enddo
    write(17,*)  " ___" 


    !write(13,*) 'dij0',pawtab(1)%dij0
contains
    subroutine generate_qgrid(gsqcut,qgrid,mqgrid)
        real*8  :: gsqcut
        real*8  :: qmax, dq
        real*8  :: qgrid(mqgrid)

        integer :: mqgrid, iq

        qmax = 1.2d0 * sqrt(gsqcut)
        dq = qmax/(1.0*(mqgrid-1))
        do iq = 1,mqgrid
            qgrid(iq) = (iq-1)*dq
        enddo
    end subroutine
end subroutine
