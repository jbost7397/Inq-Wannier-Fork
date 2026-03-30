! The input variables:
! 1. ecut, ecutpaw : kinetic energy cutoff of the planewave basis set
! there will be one coarse grid for density/potential, and a fine grid for PAW
! the unit is in Hartree
! 2. rprimd, gprimd : real and reciprocal space lattice vectors, respectively
! unit for rprimd is in Bohr, and for gprimd is in Bohr^-1
! 3. gmet : reciprocal space metric (bohr^-2)
! 4. ucvol : volume of unit cell (Bohr^3)
! 5. ngfft, ngfftdg : dimension of FFT grids of the coarse and fine grids
! 6. natom, ntypat, typat: #. atoms, #. element types
! and typat records the type of each atom
! 7. xred : coordinate of each atom, in terms of rprimd (namely, direct coordinate)
! 8. filename_list : filename of the PAW xml files for each element

subroutine fortran_main(ecut,ecutpaw,gmet,rprimd,gprimd,ucvol, &
    ngfft,ngfftdg,natom,ntypat,typat,xred,filename_list)
    use libpaw_mod, only : cplex, pawrhoij, paw_ij
    implicit none

    real*8  :: ecut, ecutpaw !a coarse grid, and a fine grid for PAW
    real*8  :: gmet(3,3) !reciprocal space metric
    real*8  :: rprimd(3,3) !lattice vectors
    real*8  :: gprimd(3,3) !reciprocal space lattice vectors
    integer :: ngfft(3),ngfftdg(3)
    real*8  :: ucvol !volume of unit cell
    integer :: ixc, xclevel !functional; will be set externally in practice
    integer :: ntypat, natom, iatom
    integer :: nfft
    integer :: nspden,nsppol
    integer :: nrhoijsel,size_rhoij
    integer :: i,j,k,l,idx
    integer, allocatable :: rhoijselect(:)
    real*8,  allocatable :: rhoijp(:,:)
    real*8,  allocatable :: vtrial(:,:), vxc(:,:)
    real*8,  allocatable :: vtrial_row_major(:,:), vxc_row_major(:,:)
    real*8,  allocatable :: vloc(:), ncoret(:)
    real*8,  allocatable :: vloc_col_major(:), ncoret_col_major(:)
    real*8,  allocatable :: vloc_row_major(:), ncoret_row_major(:)
    real*8,  allocatable :: nhat(:,:), nhatgr(:,:,:)
    real*8,  allocatable :: nhat_row_major(:,:)
    real*8,  allocatable :: dij(:,:)
    !integer, allocatable :: typat(:)
    !real*8,  allocatable :: xred(:,:)

    integer :: typat(natom)
    real*8  :: xred(3,natom)
    
    character(len=264) :: filename_list(ntypat)
    !real*8  :: epsatm(ntypat)
    real*8  :: epawdc
    real*8  :: diff_row_col_majors

    !Delete later. Debugging stuff

    integer, external :: pawrad_mesh_size
    integer, external :: pawtab_mesh_size
    integer, external :: pawxml_lmax
    integer, external :: pawxml_nchan_max
    logical, external :: pawproj_exist

    real*8, allocatable :: rad_grid(:)
    real*8, allocatable :: proj_tmp(:)
    integer :: rad_mesh_size, tab_mesh_size
    integer :: lmax, ichan, nchan

    write(*,*) '1. Setting up libpaw'

    ! (Temporary) set xc functional type
    ixc = 7 ! corresponds to PW92 LDA functional
    xclevel = 1
    nspden = 1
    nsppol = 1

    write(*,*) 'Read_proj 1'
    call prep_read_pawxml(filename_list(1))
    !rad_mesh_size = pawrad_mesh_size(filename_list(1))
    !tab_mesh_size = pawtab_mesh_size(filename_list(1))
    !allocate(rad_grid(rad_mesh_size))
    !allocate(proj_tmp(tab_mesh_size))
    !call pawrad_grid(filename_list(1),rad_mesh_size,rad_grid)
    !lmax=pawxml_lmax(filename_list(1))
    !nchan=pawxml_nchan_max(filename_list(1))
    !write(14,*) ' lmax, nchan', lmax, nchan
    !write(14,*) ' first proj ', filename_list(1)
    !write(14,*) ' mesh_size ', tab_mesh_size
    !do l=0,lmax 
    !    do ichan=1,nchan
    !        write(14,*) l, ichan, pawproj_exist(filename_list(1),l,ichan)
    !        if(pawproj_exist(filename_list(1),l,ichan)) then
    !            call pawproj(filename_list(1),l,ichan,tab_mesh_size,proj_tmp)
    !            write(14,*) l, ichan, sum(proj_tmp)
    !            write(14,*) proj_tmp(1:5)
    !            write(14,*) maxval(proj_tmp),minval(proj_tmp)
    !        endif
    !    enddo
    !enddo

    write(*,*) 'Read_proj 2'
    call prep_read_pawxml(filename_list(2))
    !lmax=pawxml_lmax(filename_list(2))
    !nchan=pawxml_nchan_max(filename_list(2))
    !write(14,*) ' lmax, nchan', lmax, nchan
    !write(14,*) ' second proj ', filename_list(2)
    !tab_mesh_size = pawtab_mesh_size(filename_list(2))
    !write(14,*) ' mesh_size ', tab_mesh_size
    !do l=0,lmax 
    !    do ichan=1,2
    !        write(14,*) l, ichan, pawproj_exist(filename_list(2),l,ichan)
    !        if(pawproj_exist(filename_list(2),l,ichan)) then
    !            call pawproj(filename_list(2),l,ichan,tab_mesh_size,proj_tmp)
    !            write(14,*) l, ichan, sum(proj_tmp)
    !            write(14,*) proj_tmp(1:5)
    !            write(14,*) maxval(proj_tmp),minval(proj_tmp)
    !        endif
    !    enddo
    !enddo

    write(*,*) 'mesh_size for ', trim(adjustl(filename_list(2))), ' is ', pawrad_mesh_size(filename_list(2))
    write(*,*) 'Read_pawxml'
    call read_pawxml_all(ecut,ecutpaw,gmet,rprimd,gprimd,ucvol,ngfft,ngfftdg, &
        natom,ntypat,typat,xred,ixc,xclevel,filename_list,nspden,nsppol)
    write(*,*) 'prepare_libpaw'
    call prepare_libpaw(ecut,ecutpaw,gmet,rprimd,gprimd,ucvol,ngfft,ngfftdg, &
        natom,ntypat,typat,xred,ixc,xclevel,filename_list,nspden,nsppol)

    nfft = ngfftdg(1) * ngfftdg(2) * ngfftdg(3)
    allocate(ncoret(nfft),vloc(nfft))
    allocate(ncoret_row_major(nfft),vloc_row_major(nfft))
    allocate(ncoret_col_major(nfft),vloc_col_major(nfft))
    allocate(vtrial(cplex*nfft,nspden),vxc(cplex*nfft,nspden))
    allocate(vtrial_row_major(cplex*nfft,nspden),vxc_row_major(cplex*nfft,nspden))
    allocate(nhat(nfft,nspden), nhatgr(nfft,nspden,3))
    allocate(nhat_row_major(nfft,nspden))

    call get_vloc_ncoret(ngfftdg,nfft,natom,ntypat,rprimd,gprimd,gmet,ucvol,xred,ncoret,vloc)

    call col_to_row_major_ordering(ngfftdg(1),ngfftdg(2),ngfftdg(3),nfft,vloc,vloc_row_major)
    call col_to_row_major_ordering(ngfftdg(1),ngfftdg(2),ngfftdg(3),nfft,ncoret,ncoret_row_major)

    open(unit=34,file='vloc.dat')
    open(unit=35,file='ncoret.dat')
    do i=1,nfft
        write(34,*) vloc_row_major(i)
        write(35,*) ncoret_row_major(i)
    enddo
    close(34); close(35)
    
    open(unit=10,file='rhoij')
    do iatom = 1, natom
        read(10,*) nrhoijsel

        size_rhoij = size(pawrhoij(iatom)%rhoijselect)
        allocate(rhoijselect(size_rhoij),rhoijp(size_rhoij,nspden))
        read(10,*) rhoijselect
        read(10,*) rhoijp
        call set_rhoij(iatom,nrhoijsel,size_rhoij,nspden,rhoijselect,rhoijp)
        deallocate(rhoijselect,rhoijp)
    enddo
    close(10)

    call get_nhat(natom,ntypat,xred,ngfft,nfft,nspden,gprimd,rprimd,ucvol,nhat,nhatgr)
    
    call col_to_row_major_ordering(ngfftdg(1),ngfftdg(2),ngfftdg(3),nfft,nhat,nhat_row_major)

    open(unit=10,file='nhat.dat')
    do i=1,nfft
        write(10,*)  nhat_row_major(i,1)
    enddo
    close(10)
    open(unit=10,file='veff')
    read(10,*) vtrial_row_major
    read(10,*) vxc_row_major
    close(10)


    !call row_to_col_major_ordering(ngfftdg(1),ngfftdg(2),ngfftdg(3),nfft,vloc_row_major,vloc_col_major)
    !call row_to_col_major_ordering(ngfftdg(1),ngfftdg(2),ngfftdg(3),nfft,ncoret_row_major,ncoret_col_major)
    !write(6,*) "test 1 vloc row,col", sum(abs(vloc_col_major-vloc))
    !write(6,*) "test 2 ncoret row,col", sum(abs(ncoret_col_major-ncoret))
    call row_to_col_major_ordering(ngfftdg(1),ngfftdg(2),ngfftdg(3),nfft,vtrial_row_major,vtrial)
    call row_to_col_major_ordering(ngfftdg(1),ngfftdg(2),ngfftdg(3),nfft,vxc_row_major,vxc)


    vtrial = vtrial / 2d0
    vxc = vxc / 2d0

    call calculate_dij(natom,ntypat,ixc,xclevel,nfft,nspden,xred,ucvol,gprimd,vtrial,vxc,epawdc)

    open(unit=15,file='dij.dat')
    do iatom = 1, natom
        size_rhoij = size(paw_ij(iatom)%dij)
        allocate(dij(size_rhoij,nspden))
        call get_dij(iatom,size_rhoij,nspden,dij)
        do i=1,size_rhoij
            write(15,*) dij(i,1)
        enddo
        deallocate(dij)
    enddo

end subroutine

subroutine col_to_row_major_ordering(nx,ny,nz,nn,v_col,v_row)
    implicit none
    integer :: nx,ny,nz,nn
    real*8  :: v_row(nn),v_col(nn)
    integer :: i,j,k
    integer :: idx, l
    if(nx*ny*nz /= nn) stop "nx*ny*nz /= nn in col_to_row"
    idx=1
    do i = 1, nx
        do j = 1, ny
            do k = 1, nz
                l = (i - 1) + (j - 1) * nx + (k - 1) * nx * ny + 1
                v_row(idx) = v_col(l)
                idx=idx+1
            end do
        end do
    end do

end subroutine

subroutine row_to_col_major_ordering(nx,ny,nz,nn,v_row,v_col)
    implicit none
    integer :: nx,ny,nz,nn
    real*8  :: v_row(nn),v_col(nn)
    integer :: i,j,k
    integer :: idx, l
    if(nx*ny*nz /= nn) stop "nx*ny*nz /= nn in row_to_col"
    idx=1
    do i = 1, nx
        do j = 1, ny
            do k = 1, nz
                l = (j - 1) * nx + (i - 1) + (k - 1) * nx * ny + 1
                v_col(l) = v_row(idx)
                idx=idx+1
            end do
        end do
    end do

end subroutine