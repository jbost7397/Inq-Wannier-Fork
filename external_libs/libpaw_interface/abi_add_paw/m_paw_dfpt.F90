!!****m* m_paw_dfpt/m_paw_dfpt
!! NAME
!!  m_paw_dfpt
!!
!! FUNCTION
!!  This module contains several routines related to the 1st and 2nd order derivatives
!!    (in the DFPT approach) of PAW on-site quantities.
!!
!! COPYRIGHT
!! Copyright (C) 2018-2022 ABINIT group (MT,AM,FJ,JWZ)
!! This file is distributed under the terms of the
!! GNU General Public License, see ~abinit/COPYING
!! or http://www.gnu.org/copyleft/gpl.txt .
!!
!! SOURCE

#include "libpaw.h"

MODULE m_paw_dfpt

 USE_DEFS
 USE_MSG_HANDLING
 USE_MPI_WRAPPERS
 USE_MEMORY_PROFILING

 use m_pawang,       only : pawang_type
 use m_pawtab,       only : pawtab_type
 use m_pawrhoij,     only : pawrhoij_type
 use m_pawfgrtab,    only : pawfgrtab_type
 use m_paw_finegrid, only : pawgylm
 use m_paral_atom,   only : get_my_atmtab,free_my_atmtab

 implicit none

 private

!public procedures.
 public :: pawgrnl       ! Compute derivatives of total energy due to NL terms (PAW Dij derivatives)

CONTAINS  !========================================================================================
!!***

!----------------------------------------------------------------------

!!****f* m_paw_dfpt/pawgrnl
!!
!! NAME
!! pawgrnl
!!
!! FUNCTION
!! PAW: Add to GRadients of total energy due to non-local term of Hamiltonian
!!      the contribution due to Dij derivatives
!! In particular, compute contribution to forces, stresses, dyn. matrix
!! Remember: Vnl=Sum_ij[|p_i>Dij<p_j|]
!!
!! INPUTS
!!  atindx1(natom)=index table for atoms, inverse of atindx
!!  dimnhat=second dimension of array nhat (0 or # of spin components)
!!  distribfft<type(distribfft_type)>=--optional-- contains all the information related
!!                                    to the FFT parallelism and plane sharing
!!  dyfr_cplex=1 if dyfrnl is real, 2 if it is complex
!!  gsqcut=Fourier cutoff on G^2 for "large sphere" of radius double that of the basis sphere
!!  mgfft=maximum size of 1D FFTs
!!  me_g0=--optional-- 1 if the current process treat the g=0 plane-wave (only needed when comm_fft is present)
!!  mpi_atmtab(:)=--optional-- indexes of the atoms treated by current proc
!!  comm_atom=--optional-- MPI communicator over atoms
!!  comm_fft=--optional-- MPI communicator over FFT components (=mpi_comm_grid is not present)
!!  mpi_comm_grid=--optional-- MPI communicator over real space grid components (=comm_fft is not present)
!!  my_natom=number of atoms treated by current processor
!!  natom=total number of atoms in cell
!!  nattyp(ntypat)=array describing how many atoms of each type in cell
!!  nfft=(effective) number of FFT grid points (for this processor)
!!  ngfft(18)=contain all needed information about 3D FFT, see ~abinit/doc/variables/vargs.htm#ngfft
!!  nhat(nfft,dimnhat)=compensation charge density on rectangular grid in real space
!!  nspden=number of spin-density components
!!  nsym=number of symmetries in space group
!!  ntypat=number of types of atoms
!!  optgr= 1 if gradients with respect to atomic position(s) have to be computed
!!  optgr2= 1 if 2nd gradients with respect to atomic position(s) have to be computed
!!  optstr= 1 if gradients with respect to strain(s) have to be computed
!!  optstr2= 1 if 2nd gradients with respect to strain(s) have to be computed
!!  paral_kgb=--optional-- 1 if "band-FFT" parallelism is activated (only needed when comm_fft is present)
!!  pawang <type(pawang_type)>=paw angular mesh and related data
!!  pawfgrtab(my_natom) <type(pawfgrtab_type)>=atomic data given on fine rectangular grid
!!  pawrhoij(my_natom) <type(pawrhoij_type)>= paw rhoij occupancies and related data
!!  pawtab(ntypat) <type(pawtab_type)>=paw tabulated starting data
!!  ph1d(2,3*(2*mgfft+1)*natom)=1-dim phase (structure factor) information
!!  psps <type(pseudopotential_type)>=variables related to pseudopotentials
!!  qphon(3)=wavevector of the phonon
!!  rprimd(3,3)=dimensional primitive translations in real space (bohr)
!!  symrec(3,3,nsym)=symmetries in reciprocal space, reduced coordinates
!!  typat(natom)=types of atoms
!!  ucvol=unit cell volume
!!  vtrial(nfft,nspden)= total local potential
!!  vxc(nfft,nspden)=XC potential
!!  xred(3,natom)=reduced dimensionless atomic coordinates
!!
!! SIDE EFFECTS
!!  At input, this terms contain contribution from non-local projectors derivatives
!!  At output, they are updated with the contribution of Dij derivatives
!!  ==== if optgr=1 ====
!!   grnl(3*natom) =gradients of NL energy wrt atomic coordinates
!!  ==== if optstr=1 ====
!!   nlstr(6) =gradients of NL energy wrt strains
!!  ==== if optgr2=1 ====
!!   dyfrnl(dyfr_cplex,3,3,natom,natom) =2nd gradients of NL energy wrt atomic coordinates
!!  ==== if optstr=2 ====
!!    eltfrnl(6+3*natom,6)=non-symmetrized non-local contribution to the elastic tensor
!! NOTES
!!   In the case of parallelisation over atoms and calculation of dynamical matrix (optgr2=1)
!!   several data are gathered and no more distributed inside this routine.
!!
!! SOURCE

subroutine pawgrnl(atindx1,dimnhat,grnl,gsqcut,mgfft,my_natom,natom,&
&          nattyp,nfft,ngfft,nhat,nlstr,nspden,ntypat,optgr,optgr2,optstr,optstr2,&
&          pawang,pawfgrtab,pawrhoij,pawtab,ph1d,qphon,rprimd,gmet,typat,ucvol,vtrial,vxc,xred,&
&          mpi_atmtab,comm_atom,comm_fft,mpi_comm_grid,me_g0,paral_kgb)

!Arguments ------------------------------------
!scalars
 integer,intent(in) :: dimnhat,mgfft,my_natom,natom,nfft,nspden,ntypat
 integer,intent(in) :: optgr,optgr2,optstr,optstr2
 integer,optional,intent(in) :: me_g0,comm_atom,comm_fft,mpi_comm_grid,paral_kgb
 real(dp),intent(in) :: gsqcut,ucvol
 type(pawang_type),intent(in) :: pawang
!arrays
 integer,intent(in) :: atindx1(natom),nattyp(ntypat),ngfft(3)
 integer,intent(in) :: typat(natom)
 integer,optional,target,intent(in) :: mpi_atmtab(:)
 real(dp),intent(in) :: nhat(nfft,dimnhat),ph1d(2,3*(2*mgfft+1)*natom),qphon(3)
 real(dp),intent(in) :: rprimd(3,3),vxc(nfft,nspden),xred(3,natom)
 real(dp),intent(in),target :: vtrial(nfft,nspden)
 real(dp),intent(inout) :: grnl(3*natom*optgr)
 real(dp),intent(inout) :: nlstr(6*optstr)
 type(pawfgrtab_type),target,intent(inout) :: pawfgrtab(:)
 type(pawrhoij_type),target,intent(inout) ::  pawrhoij(:)
 type(pawtab_type),intent(in) :: pawtab(ntypat)

!Local variables-------------------------------
!scalars
 integer :: bufind,bufsiz,cplex,dimvtrial,eps_alpha,eps_beta,eps_gamma,eps_delta,iatm,iatom
 integer :: iatom_pawfgrtab,iatom_pawrhoij,iatom_tot,iatshft,ic,idiag,idir,ier,ilm,indx,irhoij
 integer :: isel,ishift_grhoij,ishift_gr,ishift2_gr,ishift_gr2,ishift_str,ishift_str2,ishift_str2is,ispden
 integer :: ispvtr,itypat,jatom,jatom_tot,jatm,jc,jrhoij,jtypat,klm,klmn,klmn1,ll,lm_size
 integer :: lm_sizej,lmax,lmin,lmn2_size,me_fft,mu,mua,mub,mushift,my_me_g0,my_comm_atom,my_comm_fft
 integer :: my_comm_grid,my_paral_kgb,n1,n2,n3,nfftot,nfgd,nfgd_jatom
 integer :: ngrad,ngrad_nondiag,ngradp,ngradp_nondiag,ngrhat,nsploop
 integer :: opt1,opt2,opt3,qne0,usexcnhat
 logical,parameter :: save_memory=.true.
 logical :: has_phase,my_atmtab_allocated
 logical :: paral_atom,paral_atom_pawfgrtab,paral_atom_pawrhoij,paral_grid
 real(dp) :: dlt_tmp,fact_ucvol,grhat_x,hatstr_diag,rcut_jatom,ro,ro_d,ucvol_
 character(len=500) :: msg
 type(pawfgrtab_type),pointer :: pawfgrtab_iatom,pawfgrtab_jatom
 type(pawrhoij_type),pointer :: pawrhoij_iatom,pawrhoij_jatom
!arrays
 integer,parameter :: alpha(9)=(/1,2,3,3,3,2,2,1,1/),beta(9)=(/1,2,3,2,1,1,3,3,2/)
 integer,parameter :: eps1(6)=(/1,2,3,2,3,1/),eps2(6)=(/1,2,3,3,1,2/)
 integer,parameter :: mu9(9)=(/1,2,3,4,5,6,4,5,6/)
 integer,allocatable :: atindx(:),atm_indx(:),mu4(:)
 integer,allocatable,target :: ifftsph_tmp(:)
 integer, pointer :: my_atmtab(:)
 real(dp) :: gmet(3,3),gprimd(3,3),hatstr(6),rdum(1),rdum2(1),rmet(3,3),tmp(12)
 real(dp),allocatable :: buf(:,:),buf1(:),dyfr(:,:,:,:,:),eltfr(:,:)
 real(dp),allocatable :: grhat_tmp(:,:),grhat_tmp2(:,:),hatgr(:)
 real(dp),allocatable :: prod(:,:),prodp(:,:),vloc(:),vpsp1_gr(:,:),vpsp1_str(:,:)
 real(dp),LIBPAW_CONTIGUOUS pointer :: vtrial_(:,:)
 type(coeff2_type),allocatable :: prod_nondiag(:),prodp_nondiag(:)
 type(pawfgrtab_type),pointer :: pawfgrtab_(:),pawfgrtab_tot(:)
 type(pawrhoij_type),pointer :: pawrhoij_(:),pawrhoij_tot(:)

! *************************************************************************

 !DBG_ENTER("COLL")

!Compatibility tests
 qne0=0;if (qphon(1)**2+qphon(2)**2+qphon(3)**2>=1.d-15) qne0=1
 if (my_natom>0) then
   if (pawrhoij(1)%qphase/=1) then
     msg='pawgrnl: not supposed to be called with pawrhoij(:)%qphase=2!'
     ABI_BUG(msg)
   end if
 end if

!----------------------------------------------------------------------
!Parallelism setup
!Set up parallelism over atoms
 paral_atom=(present(comm_atom).and.(my_natom/=natom))
 paral_atom_pawfgrtab=(size(pawfgrtab)/=natom)
 paral_atom_pawrhoij=(size(pawrhoij)/=natom)
 nullify(my_atmtab);if (present(mpi_atmtab)) my_atmtab => mpi_atmtab
 my_comm_atom=xpaw_mpi_comm_self;if (present(comm_atom)) my_comm_atom=comm_atom
 call get_my_atmtab(my_comm_atom,my_atmtab,my_atmtab_allocated,paral_atom,natom,my_natom_ref=my_natom)
 if (paral_atom) then
   LIBPAW_ALLOCATE(atm_indx,(natom))
   atm_indx=-1
   do iatom=1,my_natom
     atm_indx(my_atmtab(iatom))=iatom
   end do
 end if

!Set up parallelism over real space grid and/or FFT
 n1=ngfft(1);n2=ngfft(2);n3=ngfft(3);nfftot=n1*n2*n3
 my_comm_grid=xpaw_mpi_comm_self;my_comm_fft=xpaw_mpi_comm_self;me_fft=0
 my_me_g0=1;my_paral_kgb=0;paral_grid=.false.;
 !if (present(mpi_comm_grid).or.present(comm_fft)) then
 !  if (present(mpi_comm_grid)) my_comm_grid=mpi_comm_grid
 !  if (present(comm_fft)) my_comm_fft=comm_fft
 !  if (.not.present(mpi_comm_grid)) my_comm_grid=comm_fft
 !  if (.not.present(comm_fft)) my_comm_fft=mpi_comm_grid
 !  paral_grid=(xmpi_comm_size(my_comm_grid)>1)
 !  me_fft=xmpi_comm_rank(my_comm_fft)
 !end if

!----------------------------------------------------------------------
!Initializations

!Compute different geometric tensors
!ucvol is not computed here but provided as input arg
 !call metric(gmet,gprimd,-1,rmet,rprimd,ucvol_)
 fact_ucvol=ucvol/dble(nfftot)

!Retrieve local potential according to the use of nhat in XC
 usexcnhat=maxval(pawtab(1:ntypat)%usexcnhat)
 if (usexcnhat==0) then
!  dimvtrial=nspden
   dimvtrial=1
   LIBPAW_ALLOCATE(vtrial_,(nfft,dimvtrial))
!$OMP PARALLEL DO PRIVATE(ic) SHARED(nfft,vtrial,vtrial_,vxc)
   do ic=1,nfft
     vtrial_(ic,1:dimvtrial)=vtrial(ic,1:dimvtrial)-vxc(ic,1:dimvtrial)
   end do
 else
   dimvtrial=nspden
   vtrial_ => vtrial
 end if

!Initializations and allocations
 ngrhat=0;ngrad=0;ngradp=0;ngrad_nondiag=0;ngradp_nondiag=0
 ishift_grhoij=0;ishift_gr=0;ishift_gr2=0;ishift_str=0;ishift_str2=0;ishift_str2is=0;ishift2_gr=0
 cplex=1;if (qne0==1) cplex=2
 if (optgr==1) then
   LIBPAW_ALLOCATE(hatgr,(3*natom))
   hatgr=zero
   ngrad=ngrad+3
   ngrhat=ngrhat+3
   ishift_gr2=ishift_gr2+3
 end if

 if (optstr==1) then
   hatstr=zero
   ngrad=ngrad+6
   ngrhat=ngrhat+6
   ishift_gr=ishift_gr+6
   ishift_gr2=ishift_gr2+6
   ishift_str2=ishift_str2+6
   ishift_str2is = ishift_str2is+6
 end if

!DEBUG
!   write(6,*)' preparatory computations : usexcnhat, nspden, dimvtrial=',usexcnhat, nspden, dimvtrial
!ENDDEBUG
!nsploop=nspden;if (dimvtrial<nspden) nsploop=2
 nsploop=nspden;if (dimvtrial<nspden .and. nspden==4) nsploop=1

   LIBPAW_ALLOCATE(grhat_tmp,(ngrhat,1))


!The computation of dynamical matrix and elastic tensor requires the knowledge of
!g_l(r-R).Y_lm(r-R) and derivatives for all atoms
!Compute them here, except memory saving is activated

!The computation of dynamical matrix and elastic tensor might require some communications

   pawfgrtab_tot => pawfgrtab
   pawrhoij_tot => pawrhoij


 if (save_memory) then
   pawfgrtab_ => pawfgrtab
   pawrhoij_  => pawrhoij
 else
   pawfgrtab_ => pawfgrtab_tot
   pawrhoij_  => pawrhoij_tot
 end if

!----------------------------------------------------------------------
!Loops over types and atoms

 iatshft=0
 do itypat=1,ntypat

   lmn2_size=pawtab(itypat)%lmn2_size
   lm_size=pawtab(itypat)%lcut_size**2

   do iatm=iatshft+1,iatshft+nattyp(itypat)

     iatom_tot=atindx1(iatm)
     iatom=iatom_tot

     if (iatom==-1) cycle
     iatom_pawfgrtab=iatom_tot;if (paral_atom_pawfgrtab) iatom_pawfgrtab=iatom
     iatom_pawrhoij =iatom_tot;if (paral_atom_pawrhoij)  iatom_pawrhoij =iatom
     pawfgrtab_iatom => pawfgrtab_(iatom_pawfgrtab)
     pawrhoij_iatom  => pawrhoij_(iatom_pawrhoij)

     idiag=1
     nfgd=pawfgrtab_iatom%nfgd

     LIBPAW_ALLOCATE(vloc,(nfgd))
     if (ngrad>0)  then
       LIBPAW_ALLOCATE(prod,(ngrad,lm_size))
     end if
     if (ngradp>0)  then
       LIBPAW_ALLOCATE(prodp,(ngradp,lm_size))
     end if
     if (ngrad_nondiag>0.and.ngradp_nondiag>0) then
       do jatm=1,natom
         jtypat=typat(atindx1(jatm))
         lm_sizej=pawtab(jtypat)%lcut_size**2
         LIBPAW_ALLOCATE(prod_nondiag(jatm)%value,(ngrad_nondiag,lm_sizej))
         LIBPAW_ALLOCATE(prodp_nondiag(jatm)%value,(ngradp_nondiag,lm_sizej))
       end do
     end if

     grhat_tmp=zero

!    ------------------------------------------------------------------
!    Compute some useful data

!    Eventually compute g_l(r).Y_lm(r) derivatives for the current atom (if not already done)
     if ((optgr==1.or.optstr==1).and.(optgr2/=1).and.(optstr2/=1)) then
       if (pawfgrtab_iatom%gylmgr_allocated==0) then
         if (allocated(pawfgrtab_iatom%gylmgr))  then
           LIBPAW_DEALLOCATE(pawfgrtab_iatom%gylmgr)
         end if
         LIBPAW_ALLOCATE(pawfgrtab_iatom%gylmgr,(3,pawfgrtab_iatom%nfgd,lm_size))
         pawfgrtab_iatom%gylmgr_allocated=2
         call pawgylm(rdum,pawfgrtab_iatom%gylmgr,rdum2,lm_size,pawfgrtab_iatom%nfgd,&
&         0,1,0,pawtab(itypat),pawfgrtab_iatom%rfgd)
       end if

     end if

!    ------------------------------------------------------------------
!    Loop over spin components

     do ispden=1,nsploop

!      ----- Retrieve potential (subtle if nspden=4 ;-)
       if (nspden/=4) then
         ispvtr=min(dimvtrial,ispden)
         do ic=1,nfgd
           jc = pawfgrtab_iatom%ifftsph(ic)
           vloc(ic)=vtrial_(jc,ispvtr)
         end do
       else
         if (ispden==1) then
           ispvtr=min(dimvtrial,2)
           do ic=1,nfgd
             jc=pawfgrtab_iatom%ifftsph(ic)
             vloc(ic)=half*(vtrial_(jc,1)+vtrial_(jc,ispvtr))
           end do
         else if (ispden==4) then
           ispvtr=min(dimvtrial,2)
           do ic=1,nfgd
             jc=pawfgrtab_iatom%ifftsph(ic)
             vloc(ic)=half*(vtrial_(jc,1)-vtrial_(jc,ispvtr))
           end do
         else if (ispden==2) then
           ispvtr=min(dimvtrial,3)
           do ic=1,nfgd
             jc=pawfgrtab_iatom%ifftsph(ic)
             vloc(ic)=vtrial_(jc,ispvtr)
           end do
         else ! ispden=3
           ispvtr=min(dimvtrial,4)
           do ic=1,nfgd
             jc=pawfgrtab_iatom%ifftsph(ic)
             vloc(ic)=-vtrial_(jc,ispvtr)
           end do
         end if
       end if

!      -----------------------------------------------------------------------
!      ----- Compute projected scalars (integrals of vloc and Q_ij^hat) ------
!      ----- and/or their derivatives ----------------------------------------

       if (ngrad>0) prod=zero
       if (ngradp>0) prodp=zero

!      ==== Contribution to forces ====
       if (optgr==1) then
         do ilm=1,lm_size
           do ic=1,pawfgrtab_iatom%nfgd
             do mu=1,3
               prod(mu+ishift_gr,ilm)=prod(mu+ishift_gr,ilm)-&
&               vloc(ic)*pawfgrtab_iatom%gylmgr(mu,ic,ilm)
             end do
           end do
         end do
       end if ! optgr

!      ==== Contribution to stresses ====
       if (optstr==1) then
         do ilm=1,lm_size
           do ic=1,pawfgrtab_iatom%nfgd
             jc=pawfgrtab_iatom%ifftsph(ic)
             do mu=1,6
               mua=alpha(mu);mub=beta(mu)
               prod(mu+ishift_str,ilm)=prod(mu+ishift_str,ilm) &
&               +half*vloc(ic)&
&               *(pawfgrtab_iatom%gylmgr(mua,ic,ilm)*pawfgrtab_iatom%rfgd(mub,ic)&
&               +pawfgrtab_iatom%gylmgr(mub,ic,ilm)*pawfgrtab_iatom%rfgd(mua,ic))
             end do
           end do
         end do
       end if ! optstr
!DEBUG
!   write(6,*)' after loops on ilm, ic, mu : ispden,lm_size, pawfgrtab_iatom%nfgd=',ispden,lm_size, pawfgrtab_iatom%nfgd
!   write(6,*)' after loops on ilm, ic, mu, writes ilm, prod(1+ishift_str,ilm:lm_size) when bigger than tol10 (ilm between 1 and lm_size)'
!   do ilm=1, lm_size
!     if( abs(prod(1+ishift_str,ilm))>tol6 )then
!       write(6,*)ilm,prod(1+ishift_str,ilm)
!     endif
!   enddo
!ENDDEBUG

!      --- Apply scaling factor on integrals ---
       if (ngrad >0) prod (:,:)=prod (:,:)*fact_ucvol
       if (ngradp>0) prodp(:,:)=prodp(:,:)*fact_ucvol
       if (ngrad_nondiag>0) then
         do jatm=1,natom
           prod_nondiag(jatm)%value(:,:)=prod_nondiag(jatm)%value(:,:)*fact_ucvol
         end do
       end if
       if (ngradp_nondiag>0) then
         do jatm=1,natom
           prodp_nondiag(jatm)%value(:,:)=prodp_nondiag(jatm)%value(:,:)*fact_ucvol
         end do
       end if

!      --- Reduction in case of parallelization ---
!        if (paral_grid) then
!          if (ngrad>0) then
!            call xpaw_mpi_sum(prod,my_comm_grid,ier)
!          end if
!          if (ngradp>0) then
!            call xpaw_mpi_sum(prodp,my_comm_grid,ier)
!          end if
!          if (ngrad_nondiag>0.or.ngradp_nondiag>0) then
!            bufsiz=0;bufind=0
!            do jatm=1,natom
!              jtypat=typat(atindx1(jatm))
!              bufsiz=bufsiz+pawtab(jtypat)%lcut_size**2
!            end do
!            LIBPAW_ALLOCATE(buf,(ngrad_nondiag+ngradp_nondiag,bufsiz))
!            do jatm=1,natom
!              jtypat=typat(atindx1(jatm))
!              lm_sizej=pawtab(jtypat)%lcut_size**2
!              if (ngrad_nondiag> 0) buf(1:ngrad_nondiag,bufind+1:bufind+lm_sizej)= &
! &             prod_nondiag(jatm)%value(:,:)
!              if (ngradp_nondiag>0) buf(ngrad_nondiag+1:ngrad_nondiag+ngradp_nondiag, &
! &             bufind+1:bufind+lm_sizej)=prodp_nondiag(jatm)%value(:,:)
!              bufind=bufind+lm_sizej*(ngrad_nondiag+ngradp_nondiag)
!            end do
!            call xpaw_mpi_sum(buf,my_comm_grid,ier)
!            bufind=0
!            do jatm=1,natom
!              jtypat=typat(atindx1(jatm))
!              lm_sizej=pawtab(jtypat)%lcut_size**2
!              if (ngrad> 0) prod_nondiag(jatm)%value(:,:)= &
! &             buf(1:ngrad_nondiag,bufind+1:bufind+lm_sizej)
!              if (ngradp>0) prodp_nondiag(jatm)%value(:,:)= &
! &             buf(ngrad_nondiag+1:ngrad_nondiag+ngradp_nondiag,bufind+1:bufind+lm_sizej)
!              bufind=bufind+lm_sizej*(ngrad_nondiag+ngradp_nondiag)
!            end do
!            LIBPAW_DEALLOCATE(buf)
!          end if
!        end if

!      ----------------------------------------------------------------
!      Compute final sums (i.e. derivatives of Sum_ij[rho_ij.Intg{Qij.Vloc}]

!      ---- Compute terms common to all gradients
       jrhoij=1
       do irhoij=1,pawrhoij_iatom%nrhoijsel
         klmn=pawrhoij_iatom%rhoijselect(irhoij)
         klm =pawtab(itypat)%indklmn(1,klmn)
         lmin=pawtab(itypat)%indklmn(3,klmn)
         lmax=pawtab(itypat)%indklmn(4,klmn)
         ro =pawrhoij_iatom%rhoijp(jrhoij,ispden)
         ro_d=ro*pawtab(itypat)%dltij(klmn)
         do ll=lmin,lmax,2
           do ilm=ll**2+1,(ll+1)**2
             isel=pawang%gntselect(ilm,klm)
             if (isel>0) then
               grhat_x=ro_d*pawtab(itypat)%qijl(ilm,klmn)
               do mu=1,ngrad
                 grhat_tmp(mu,idiag)=grhat_tmp(mu,idiag)+grhat_x*prod(mu,ilm)
! DEBUG
!               if(mu==ishift_str+1 .and. &
!&                  (abs(grhat_x*prod(mu,ilm))>tol6 .or. irhoij==1 )      )then
!                 write(6,'(a,5i4,3es16.6)')&
!&                  'mu,idiag,ilm,irhoij,ll, grhat_tmp(mu,idiag),grhat_x,prod(mu,ilm)=',&
!&                   mu,idiag,ilm,irhoij,ll, grhat_tmp(mu,idiag),grhat_x,prod(mu,ilm)
!               endif
! ENDDEBUG
               end do
             end if
           end do
         end do
         jrhoij=jrhoij+pawrhoij_iatom%cplex_rhoij
       end do

! DEBUG
!      write(6,*)' Accumulation of grhat_tmp : idiag,grhat_tmp(ishift_str+1,idiag),',idiag,grhat_tmp(ishift_str+1,idiag)
! ENDDEBUG

!    ----------------------------------------------------------------
!    End of loop over spin components

     end do ! ispden

!    Eventually free temporary space for g_l(r).Y_lm(r) factors
     if (pawfgrtab_iatom%gylm_allocated==2) then
       LIBPAW_DEALLOCATE(pawfgrtab_iatom%gylm)
       LIBPAW_ALLOCATE(pawfgrtab_iatom%gylm,(0,0))
       pawfgrtab_iatom%gylm_allocated=0
     end if
     if (pawfgrtab_iatom%gylmgr_allocated==2) then
       LIBPAW_DEALLOCATE(pawfgrtab_iatom%gylmgr)
       LIBPAW_ALLOCATE(pawfgrtab_iatom%gylmgr,(0,0,0))
       pawfgrtab_iatom%gylmgr_allocated=0
     end if
     if (pawfgrtab_iatom%gylmgr2_allocated==2) then
       LIBPAW_DEALLOCATE(pawfgrtab_iatom%gylmgr2)
       LIBPAW_ALLOCATE(pawfgrtab_iatom%gylmgr2,(0,0,0))
       pawfgrtab_iatom%gylmgr2_allocated=0
     end if
     if (pawfgrtab_iatom%expiqr_allocated==2) then
       LIBPAW_DEALLOCATE(pawfgrtab_iatom%expiqr)
       LIBPAW_ALLOCATE(pawfgrtab_iatom%expiqr,(0,0))
       pawfgrtab_iatom%expiqr_allocated=0
     end if

!    ----------------------------------------------------------------
!    Copy results in corresponding arrays

!    ==== Forces ====
!    Convert from cartesian to reduced coordinates
     if (optgr==1) then
       mushift=3*(iatm-1)
       tmp(1:3)=grhat_tmp(ishift_gr+1:ishift_gr+3,idiag)
       do mu=1,3
         hatgr(mu+mushift)=rprimd(1,mu)*tmp(1)+rprimd(2,mu)*tmp(2)+rprimd(3,mu)*tmp(3)
       end do
     end if

!    ==== Stresses ====
     if (optstr==1) then
!      This is contribution Eq.(41) of Torrent2008.
!DEBUG
!   write(6,*)' after loop on ispden,ishift_str,idiag,hatstr(1),grhat_tmp(ishift_str+1)',hatstr(1),grhat_tmp(ishift_str+1,idiag)
!ENDDEBUG
       hatstr(1:6)=hatstr(1:6)+grhat_tmp(ishift_str+1:ishift_str+6,idiag)
     end if
!    ----------------------------------------------------------------
!    End loops on types and atoms

     LIBPAW_DEALLOCATE(vloc)
     if (ngrad>0)  then
       LIBPAW_DEALLOCATE(prod)
     end if
     if (ngradp>0)  then
       LIBPAW_DEALLOCATE(prodp)
     end if
   end do ! iatm
   iatshft=iatshft+nattyp(itypat)
 end do ! itypat

!DEBUG
!  write(6,*)' before parallelization over atoms hatstr(1)',hatstr(1)
!ENDDEBUG

!Reduction in case of parallelisation over atoms
!  if (paral_atom) then
!    bufsiz=3*natom*optgr+6*optstr
!    if (bufsiz>0) then
!      LIBPAW_ALLOCATE(buf1,(bufsiz))
!      if (optgr==1) buf1(1:3*natom)=hatgr(1:3*natom)
!      indx=optgr*3*natom
!      if (optstr==1) buf1(indx+1:indx+6)=hatstr(1:6)
!      indx=indx+optstr*6

!      call xpaw_mpi_sum(buf1,my_comm_atom,ier)
!      if (optgr==1) hatgr(1:3*natom)=buf1(1:3*natom)
!      indx=optgr*3*natom
!      if (optstr==1) hatstr(1:6)=buf1(indx+1:indx+6)
!      indx=indx+optstr*6
!      LIBPAW_DEALLOCATE(buf1)
!    end if
!  end if

!Deallocate additional memory
 LIBPAW_DEALLOCATE(grhat_tmp)

!----------------------------------------------------------------------
!Update non-local gradients

!===== Update forces =====
 if (optgr==1) then
   grnl(1:3*natom)=grnl(1:3*natom)+hatgr(1:3*natom)
   LIBPAW_DEALLOCATE(hatgr)
 end if

!===== Convert stresses (add diag and off-diag contributions) =====
 if (optstr==1) then

!  Has to compute int[nhat*vtrial]. See Eq.(40) in Torrent2008 .
   hatstr_diag=zero
   if (nspden==1.or.dimvtrial==1) then
     do ic=1,nfft
       hatstr_diag=hatstr_diag+vtrial_(ic,1)*nhat(ic,1)
     end do
   else if (nspden==2) then
     do ic=1,nfft
       hatstr_diag=hatstr_diag+vtrial_(ic,1)*nhat(ic,2)+vtrial_(ic,2)*(nhat(ic,1)-nhat(ic,2))
     end do
   else if (nspden==4) then
     do ic=1,nfft
       hatstr_diag=hatstr_diag+half*(vtrial_(ic,1)*(nhat(ic,1)+nhat(ic,4)) &
&       +vtrial_(ic,2)*(nhat(ic,1)-nhat(ic,4))) &
&       +vtrial_(ic,3)*nhat(ic,2)-vtrial_(ic,4)*nhat(ic,3)
     end do
   end if
   hatstr_diag=hatstr_diag*fact_ucvol
  !  if (paral_grid) then
  !    call xpaw_mpi_sum(hatstr_diag,my_comm_grid,ier)
  !  end if

!  Convert hat contribution

!DEBUG
!  write(6,*)' hatstr(1),hatstr_diag,nlstr(1)=',hatstr(1),hatstr_diag,nlstr(1)
!ENDDEBUG

   hatstr(1:3)=(hatstr(1:3)+hatstr_diag)/ucvol
   hatstr(4:6)= hatstr(4:6)/ucvol

!  Add to already computed NL contrib
   nlstr(1:6)=nlstr(1:6)+hatstr(1:6)

!  Apply symmetries
   !call stresssym(gprimd,nsym,nlstr,symrec)
 end if

!----------------------------------------------------------------------
!End

!Destroy temporary space
 if (usexcnhat==0)  then
   LIBPAW_DEALLOCATE(vtrial_)
 end if

!Destroy atom tables used for parallelism
 call free_my_atmtab(my_atmtab,my_atmtab_allocated)
 if (paral_atom) then
   LIBPAW_DEALLOCATE(atm_indx)
 end if

 !DBG_ENTER("COLL")

 CONTAINS
!!***

! ------------------------------------------------
!!****f* pawgrnl/pawgrnl_convert
!! NAME
!!  pawgrnl_convert
!!
!! FUNCTION
!!  notation: Convert index of the elastic tensor:
!!    - voigt notation       => 32
!!    - normal notation      => 3 3 2 2
!!    - notation for gylmgr2 => 32 32 32 32 => 4 4 4
!!
!! INPUTS
!!  eps_alpha, eps_beta, eps_delta, eps_gamma
!!
!! OUTPUT
!!  mu4(4) = array with index for the second derivative of gylm
!!
!! SIDE EFFECTS
!!  mu4(4) = input : array with index for the second derivative of gylm
!!           output: the 4 indexes for the calculation of the second derivative of gylm
!!
!! SOURCE

subroutine pawgrnl_convert(mu4,eps_alpha,eps_beta,eps_gamma,eps_delta)

!Arguments ------------------------------------
 !scalar
 integer,intent(in)  :: eps_alpha,eps_beta
 integer,optional,intent(in)  :: eps_gamma,eps_delta
 !array
 integer,intent(inout) :: mu4(4)

!Local variables-------------------------------
 integer :: eps1,eps2,i,j,k
 integer,allocatable :: mu_temp(:)

! *************************************************************************

 LIBPAW_ALLOCATE(mu_temp,(4))
 if (present(eps_gamma).and.present(eps_delta)) then
   mu_temp(1)=eps_alpha
   mu_temp(2)=eps_beta
   mu_temp(3)=eps_gamma
   mu_temp(4)=eps_delta
 else
   mu_temp(1)=eps_alpha
   mu_temp(2)=eps_beta
   mu_temp(3)= 0
   mu_temp(4)= 0
 end if
 k=1
 do i=1,2
   eps1=mu_temp(i)
   do j=1,2
     eps2=mu_temp(2+j)
     if(eps1==eps2) then
       if(eps1==1) mu4(k)=1;
       if(eps1==2) mu4(k)=2;
       if(eps1==3) mu4(k)=3;
     else
       if((eps1==3.and.eps2==2).or.(eps1==2.and.eps2==3)) mu4(k)=4;
       if((eps1==3.and.eps2==1).or.(eps1==1.and.eps2==3)) mu4(k)=5;
       if((eps1==1.and.eps2==2).or.(eps1==2.and.eps2==1)) mu4(k)=6;
     end if
     k=k+1
   end do
 end do
 LIBPAW_DEALLOCATE(mu_temp)

end subroutine pawgrnl_convert
! ------------------------------------------------

end subroutine pawgrnl
!!***

!----------------------------------------------------------------------

END MODULE m_paw_dfpt
!!***
