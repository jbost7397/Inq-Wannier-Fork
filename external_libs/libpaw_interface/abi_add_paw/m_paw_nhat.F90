!!****m* ABINIT/m_paw_nhat
!! NAME
!!  m_paw_nhat
!!
!! FUNCTION
!!  This module contains several routines related to the PAW compensation
!!    charge density (i.e. n^hat(r)).
!!
!! COPYRIGHT
!! Copyright (C) 2018-2022 ABINIT group (FJ, MT, MG, TRangel)
!! This file is distributed under the terms of the
!! GNU General Public License, see ~abinit/COPYING
!! or http://www.gnu.org/copyleft/gpl.txt .
!!
!! SOURCE

#include "libpaw.h"

MODULE m_paw_nhat

 USE_DEFS
 USE_MSG_HANDLING
 USE_MPI_WRAPPERS
 USE_MEMORY_PROFILING

 !use m_time,         only : timab
 use m_pawang,       only : pawang_type
 use m_pawtab,       only : pawtab_type
 use m_pawfgrtab,    only : pawfgrtab_type
 use m_pawrhoij,     only : pawrhoij_type
 use m_paw_finegrid, only : pawgylm,pawrfgd_fft
 use m_paral_atom,   only : get_my_atmtab, free_my_atmtab
 use m_paw_finegrid, only : pawexpiqr
 use m_cgtools,      only : mean_fftr
 !use m_distribfft,   only : distribfft_type

 implicit none

 private

!public procedures.
 public :: pawmknhat        ! Compute compensation charge density on the real space (fine) grid
 public :: nhatgrid         ! Determine points of the (fine) grid that are located around atoms - PW version

CONTAINS  !========================================================================================
!!***

!!****f* m_paw_nhat/pawmknhat
!! NAME
!! pawmknhat
!!
!! FUNCTION
!! PAW only:
!! Compute compensation charge density (and derivatives) on the fine FFT grid
!! Can also compute first-order compensation charge density (RF calculations)
!!
!! INPUTS
!!  cplex: if 1, real space 1-order functions on FFT grid are REAL, if 2, COMPLEX
!!  distribfft<type(distribfft_type)>=--optional-- contains infos related to FFT parallelism
!!  gprimd(3,3)=dimensional primitive translations for reciprocal space
!!  ider= 0: nhat(r) is computed
!!        1: cartesian derivatives of nhat(r) are computed
!!        2: nhat(r) and derivatives are computed
!!  idir=direction of atomic displacement (in case of atomic displ. perturb.)
!!  ipert=index of perturbation; must be 0 for ground-state calculations
!!  izero=if 1, unbalanced components of nhat(g) have to be set to zero
!!  Note : is actually not used during ground state calculations
!!  me_g0=--optional-- 1 if the current process treat the g=0 plane-wave (only needed when comm_fft is present)
!!  mpi_atmtab(:)=--optional-- indexes of the atoms treated by current proc
!!  comm_atom=--optional-- MPI communicator over atoms
!!  comm_fft=--optional-- MPI communicator over FFT components
!!  my_natom=number of atoms treated by current processor
!!  natom=total number of atoms in cell
!!  nfft=number of point on the rectangular fft grid
!!  ngfft(18)=contain all needed information about 3D FFT, see ~abinit/doc/variables/vargs.htm#ngfft
!!  nhatgrdim= -PAW only- 0 if pawgrnhat array is not used ; 1 otherwise
!!  ntypat=number of types of atoms in unit cell.
!!  paral_kgb=--optional-- 1 if "band-FFT" parallelism is activated (only needed when comm_fft is present)
!!  pawang <type(pawang_type)>=paw angular mesh and related data
!!  pawfgrtab(my_natom) <type(pawfgrtab_type)>=atomic data given on fine rectangular grid
!!  pawrhoij(my_natom) <type(pawrhoij_type)>= paw rhoij occupancies and related data
!!                                         (1st-order occupancies if ipert>0)
!!  pawrhoij0(my_natom) <type(pawrhoij_type)>= GS paw rhoij occupancies and related data (used only if ipert>0)
!!                                          set equal to pawrhoij for GS calculations
!!  pawtab(ntypat) <type(pawtab_type)>=paw tabulated starting data
!!  qphon(3)=wavevector of the phonon (RF only)
!!  rprimd(3,3)=dimensional primitive translations for real space
!!  ucvol=volume of the unit cell
!!  xred(3,natom)= reduced atomic coordinates
!!
!! OUTPUT
!!  === if ider=0 or 2
!!    compch_fft=compensation charge inside spheres computed over fine fft grid
!!    pawnhat(nfft,ispden)=nhat on fine rectangular grid
!!  === if ider=1 or 2
!!    pawgrnhat(nfft,ispden,3)=derivatives of nhat on fine rectangular grid (and derivatives)
!!
!! SOURCE

subroutine pawmknhat(compch_fft,cplex,ider,idir,ipert,izero,gprimd,&
&          my_natom,natom,nfft,ngfft,nhatgrdim,nspden,ntypat,pawang,pawfgrtab,&
&          pawgrnhat,pawnhat,pawrhoij,pawrhoij0,pawtab,qphon,rprimd,ucvol,usewvl,xred,&
&          mpi_atmtab,comm_atom,comm_fft,mpi_comm_wvl,me_g0,paral_kgb)!,distribfft) ! optional arguments

 implicit none

!Arguments ---------------------------------------------
!scalars
 integer,intent(in) :: cplex,ider,idir,ipert,izero,my_natom,natom,nfft
 integer,intent(in)  :: usewvl
 integer,intent(in) :: nhatgrdim,nspden,ntypat
 integer,optional,intent(in) :: me_g0,comm_atom,comm_fft,mpi_comm_wvl,paral_kgb
 real(dp),intent(in) :: ucvol
 real(dp),intent(inout) :: compch_fft
 !type(distribfft_type),optional,intent(in),target :: distribfft
 type(pawang_type),intent(in) :: pawang
!arrays
 integer,intent(in) :: ngfft(3)
 integer,optional,target,intent(in) :: mpi_atmtab(:)
 real(dp),intent(in) :: gprimd(3,3),qphon(3),rprimd(3,3),xred(3,natom)
 real(dp),intent(out) :: pawgrnhat(cplex*nfft,nspden,3*nhatgrdim)
 real(dp),intent(inout) :: pawnhat(cplex*nfft,nspden) !vz_i
 type(pawfgrtab_type),intent(inout) :: pawfgrtab(my_natom)
 type(pawrhoij_type),intent(in) :: pawrhoij(my_natom),pawrhoij0(my_natom)
 type(pawtab_type),intent(in) :: pawtab(ntypat)

!Local variables ---------------------------------------
!scalars
 integer :: cplex_rhoij,iatom,iatom_tot,ic,ierr,ii,ils,ilslm,iq0,irhoij,ispden,itypat
 integer :: jc,jrhoij,kc,klm,klmn,lmax,lmin,lm_size,mfgd,mm,mpi_comm_sphgrid
 integer :: my_comm_atom,my_comm_fft,nfgd,nfftot,option,optgr0,optgr1,optgr2,paral_kgb_fft
 logical :: compute_grad,compute_nhat,my_atmtab_allocated,need_frozen,paral_atom,qeq0
 logical :: compute_phonons,has_phase
 !type(distribfft_type),pointer :: my_distribfft !not used
 !type(mpi_type) :: mpi_enreg_fft
!arrays
 integer,pointer :: my_atmtab(:)
 real(dp) :: ro(cplex),ro_ql(cplex),tmp_compch_fft(nspden),tsec(2)
 real(dp),allocatable :: pawgrnhat_atm(:,:),pawnhat_atm(:),work(:,:)

! *************************************************************************

 !DBG_ENTER("COLL")

 compute_nhat=(ider==0.or.ider==2)
 compute_grad=(ider==1.or.ider==2)
 compute_phonons=(ipert>0.and.ipert<=natom)

!Compatibility tests
 qeq0=(qphon(1)**2+qphon(2)**2+qphon(3)**2<1.d-15)
 if (present(comm_fft)) then
   if ((.not.present(paral_kgb)).or.(.not.present(me_g0))) then
     LIBPAW_BUG('Need paral_kgb and me_g0 with comm_fft !')
   end if
 end if
 if(ider>0.and.nhatgrdim==0) then
   LIBPAW_BUG('Gradients of nhat required but not allocated!')
 end if
 if (my_natom>0) then
   if(nspden>1.and.nspden/=pawrhoij(1)%nspden) then
     LIBPAW_BUG('Wrong values for nspden and pawrhoij%nspden!')
   end if
   if(nspden>1.and.nspden/=pawfgrtab(1)%nspden) then
     LIBPAW_BUG('Wrong values for nspden and pawfgrtab%nspden!')
   end if
   if(pawrhoij(1)%qphase<cplex) then
     LIBPAW_BUG('Must have pawrhoij()%qphase >= cplex!')
   end if
   if (compute_phonons.and.(.not.qeq0)) then
     if (pawfgrtab(1)%rfgd_allocated==0) then
       LIBPAW_BUG('pawfgrtab()%rfgd array must be allocated!')
     end if
     if (compute_grad.and.(.not.compute_nhat)) then
       LIBPAW_BUG('When q<>0, nhat gradients need nhat!')
     end if
   end if
 end if

!nhat1 does not have to be computed for ddk or d2dk
 if (ipert==natom+1.or.ipert==natom+10) return

!Set up parallelism over atoms
 paral_atom=(present(comm_atom).and.(my_natom/=natom))
 nullify(my_atmtab);if (present(mpi_atmtab)) my_atmtab => mpi_atmtab
 my_comm_atom=xpaw_mpi_comm_self;if (present(comm_atom)) my_comm_atom=comm_atom
 call get_my_atmtab(my_comm_atom,my_atmtab,my_atmtab_allocated,paral_atom,natom,&
& my_natom_ref=my_natom)

!Initialisations
 if ((.not.compute_nhat).and.(.not.compute_grad)) return
 mfgd=zero;if (my_natom>0) mfgd=maxval(pawfgrtab(1:my_natom)%nfgd)
 if (compute_nhat) then
   LIBPAW_ALLOCATE(pawnhat_atm,(cplex*mfgd))
   pawnhat=zero
 end if
 if (compute_grad) then
   LIBPAW_ALLOCATE(pawgrnhat_atm,(cplex*mfgd,3))
   pawgrnhat=zero
 end if

!mpi communicators for spherical grid:
 mpi_comm_sphgrid=xpaw_mpi_comm_self !no communicators passed
 if(present(comm_fft) .and. usewvl==0) mpi_comm_sphgrid=comm_fft
 if(present(mpi_comm_wvl) .and. usewvl==1) mpi_comm_sphgrid=mpi_comm_wvl

!------------------------------------------------------------------------
!----- Loop over atoms
!------------------------------------------------------------------------

 do iatom=1,my_natom
   iatom_tot=iatom;if (paral_atom) iatom_tot=my_atmtab(iatom)

   itypat=pawrhoij(iatom)%itypat
   lm_size=pawfgrtab(iatom)%l_size**2
   need_frozen=((compute_nhat).and.(ipert==iatom_tot.or.ipert==natom+3.or.ipert==natom+4))
   nfgd=pawfgrtab(iatom)%nfgd
   cplex_rhoij=pawrhoij(iatom)%cplex_rhoij
   iq0=cplex_rhoij*pawrhoij(iatom)%lmn2_size

!  Eventually compute g_l(r).Y_lm(r) factors for the current atom (if not already done)
   if (((compute_nhat).and.(pawfgrtab(iatom)%gylm_allocated==0)).or.&
&   ((compute_grad).and.(pawfgrtab(iatom)%gylmgr_allocated==0)).or.&
&   ((compute_grad.and.need_frozen).and.(pawfgrtab(iatom)%gylmgr2_allocated==0))) then
     optgr0=0;optgr1=0;optgr2=0
     if ((compute_nhat).and.(pawfgrtab(iatom)%gylm_allocated==0)) then
       if (allocated(pawfgrtab(iatom)%gylm))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylm)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylm,(nfgd,pawfgrtab(iatom)%l_size**2))
       pawfgrtab(iatom)%gylm_allocated=2;optgr0=1
     end if
     if ((compute_grad).and.(pawfgrtab(iatom)%gylmgr_allocated==0)) then
       if (allocated(pawfgrtab(iatom)%gylmgr))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr,(3,nfgd,pawfgrtab(iatom)%l_size**2))
       pawfgrtab(iatom)%gylmgr_allocated=2;optgr1=1
     end if
     if ((compute_grad.and.need_frozen).and.(pawfgrtab(iatom)%gylmgr2_allocated==0)) then
       if (allocated(pawfgrtab(iatom)%gylmgr2))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr2)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr2,(6,nfgd,pawfgrtab(iatom)%l_size**2))
       pawfgrtab(iatom)%gylmgr2_allocated=2;optgr2=1
     end if
     if (optgr0+optgr1+optgr2>0) then
       call pawgylm(pawfgrtab(iatom)%gylm,pawfgrtab(iatom)%gylmgr,pawfgrtab(iatom)%gylmgr2,&
&       lm_size,nfgd,optgr0,optgr1,optgr2,pawtab(itypat),pawfgrtab(iatom)%rfgd)
     end if
   end if


!  Eventually compute exp(-i.q.r) factors for the current atom (if not already done)
   if (compute_phonons.and.(.not.qeq0).and.pawfgrtab(iatom)%expiqr_allocated==0) then
     if (allocated(pawfgrtab(iatom)%expiqr))  then
       LIBPAW_DEALLOCATE(pawfgrtab(iatom)%expiqr)
     end if
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%expiqr,(2,nfgd))
     call pawexpiqr(pawfgrtab(iatom)%expiqr,gprimd,nfgd,qphon,&
&     pawfgrtab(iatom)%rfgd,xred(:,iatom_tot))
     pawfgrtab(iatom)%expiqr_allocated=2
   end if
   has_phase=(compute_phonons.and.pawfgrtab(iatom)%expiqr_allocated/=0)

!  Eventually compute frozen part of nhat for the current atom (if not already done)
   if ((need_frozen).and.((pawfgrtab(iatom)%nhatfr_allocated==0).or.&
&   (compute_grad.and.pawfgrtab(iatom)%nhatfrgr_allocated==0))) then
     if (allocated(pawfgrtab(iatom)%nhatfr))  then
       LIBPAW_DEALLOCATE(pawfgrtab(iatom)%nhatfr)
     end if
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%nhatfr,(nfgd,pawfgrtab(iatom)%nspden))
     option=0;pawfgrtab(iatom)%nhatfr_allocated=2
     if (compute_grad) then
       option=1
       if (allocated(pawfgrtab(iatom)%nhatfrgr))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%nhatfrgr)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%nhatfrgr,(3,nfgd,pawfgrtab(iatom)%nspden))
       pawfgrtab(iatom)%nhatfrgr_allocated=2
     end if
     call pawnhatfr(option,idir,ipert,1,natom,nspden,ntypat,pawang,pawfgrtab(iatom),&
&                   pawrhoij0(iatom),pawtab,rprimd)
   end if

!  ------------------------------------------------------------------------
!  ----- Loop over density components
!  ------------------------------------------------------------------------

   do ispden=1,nspden

     if (compute_nhat) pawnhat_atm(1:cplex*nfgd)=zero
     if (compute_grad) pawgrnhat_atm(1:cplex*nfgd,1:3)=zero

!    ------------------------------------------------------------------------
!    ----- Loop over ij channels (basis components)
!    ------------------------------------------------------------------------
     jrhoij=1
     do irhoij=1,pawrhoij(iatom)%nrhoijsel
       klmn=pawrhoij(iatom)%rhoijselect(irhoij)
       klm =pawtab(itypat)%indklmn(1,klmn)
       lmin=pawtab(itypat)%indklmn(3,klmn)
       lmax=pawtab(itypat)%indklmn(4,klmn)

!      Retrieve rhoij
       if (pawrhoij(iatom)%nspden/=2) then
         ro(1)=pawrhoij(iatom)%rhoijp(jrhoij,ispden)
         if (cplex==2) ro(2)=pawrhoij(iatom)%rhoijp(iq0+jrhoij,ispden)
       else
         if (ispden==1) then
           ro(1)=pawrhoij(iatom)%rhoijp(jrhoij,1)+pawrhoij(iatom)%rhoijp(jrhoij,2)
           if (cplex==2) ro(2)=pawrhoij(iatom)%rhoijp(iq0+jrhoij,1)+pawrhoij(iatom)%rhoijp(iq0+jrhoij,2)
         else if (ispden==2) then
           ro(1)=pawrhoij(iatom)%rhoijp(jrhoij,1)
           if (cplex==2) ro(2)=pawrhoij(iatom)%rhoijp(iq0+jrhoij,1)
         end if
       end if
       ro(1:cplex)=pawtab(itypat)%dltij(klmn)*ro(1:cplex)

       if (compute_nhat) then
         if (cplex==1) then
           do ils=lmin,lmax,2
             do mm=-ils,ils
               ilslm=ils*ils+ils+mm+1
               if (pawang%gntselect(ilslm,klm)>0) then
                 ro_ql(1)=ro(1)*pawtab(itypat)%qijl(ilslm,klmn)
                 do ic=1,nfgd
                   pawnhat_atm(ic)=pawnhat_atm(ic)+ro_ql(1)*pawfgrtab(iatom)%gylm(ic,ilslm)
                 end do
               end if
             end do
           end do
         else
           do ils=lmin,lmax,2
             do mm=-ils,ils
               ilslm=ils*ils+ils+mm+1
               if (pawang%gntselect(ilslm,klm)>0) then
                 ro_ql(1:2)=ro(1:2)*pawtab(itypat)%qijl(ilslm,klmn)
                 do ic=1,nfgd
                   jc=2*ic-1
                   pawnhat_atm(jc:jc+1)=pawnhat_atm(jc:jc+1)+ro_ql(1:2)*pawfgrtab(iatom)%gylm(ic,ilslm)
                 end do
               end if
             end do
           end do
         end if
       end if

       if (compute_grad) then
         if (cplex==1) then
           do ils=lmin,lmax,2
             do mm=-ils,ils
               ilslm=ils*ils+ils+mm+1
               if (pawang%gntselect(ilslm,klm)>0) then
                 ro_ql(1)=ro(1)*pawtab(itypat)%qijl(ilslm,klmn)
                 do ic=1,nfgd
                   pawgrnhat_atm(ic,1)=pawgrnhat_atm(ic,1)+ro_ql(1)*pawfgrtab(iatom)%gylmgr(1,ic,ilslm)
                   pawgrnhat_atm(ic,2)=pawgrnhat_atm(ic,2)+ro_ql(1)*pawfgrtab(iatom)%gylmgr(2,ic,ilslm)
                   pawgrnhat_atm(ic,3)=pawgrnhat_atm(ic,3)+ro_ql(1)*pawfgrtab(iatom)%gylmgr(3,ic,ilslm)
                 end do
               end if
             end do
           end do
         else
           do ils=lmin,lmax,2
             do mm=-ils,ils
               ilslm=ils*ils+ils+mm+1
               if (pawang%gntselect(ilslm,klm)>0) then
                 ro_ql(1:2)=ro(1:2)*pawtab(itypat)%qijl(ilslm,klmn)
                 do ic=1,nfgd
                   jc=2*ic-1
                   pawgrnhat_atm(jc:jc+1,1)=pawgrnhat_atm(jc:jc+1,1) &
&                   +ro_ql(1:2)*pawfgrtab(iatom)%gylmgr(1,ic,ilslm)
                   pawgrnhat_atm(jc:jc+1,2)=pawgrnhat_atm(jc:jc+1,2) &
&                   +ro_ql(1:2)*pawfgrtab(iatom)%gylmgr(2,ic,ilslm)
                   pawgrnhat_atm(jc:jc+1,3)=pawgrnhat_atm(jc:jc+1,3) &
&                   +ro_ql(1:2)*pawfgrtab(iatom)%gylmgr(3,ic,ilslm)
                 end do
               end if
             end do
           end do
         end if
       end if

!      ------------------------------------------------------------------------
!      ----- End loop over ij channels
!      ------------------------------------------------------------------------
       jrhoij=jrhoij+cplex_rhoij
     end do

!    If RF calculation, add frozen part of 1st-order compensation density
     if (need_frozen) then
       if (cplex==1) then
         do ic=1,nfgd
           pawnhat_atm(ic)=pawnhat_atm(ic)+pawfgrtab(iatom)%nhatfr(ic,ispden)
         end do
       else
         do ic=1,nfgd
           jc=2*ic-1
           pawnhat_atm(jc)=pawnhat_atm(jc)+pawfgrtab(iatom)%nhatfr(ic,ispden)
         end do
       end if
       if (compute_grad) then
         if (cplex==1) then
           do ic=1,nfgd
             pawgrnhat_atm(ic,1)=pawgrnhat_atm(ic,1)+pawfgrtab(iatom)%nhatfrgr(1,ic,ispden)
             pawgrnhat_atm(ic,2)=pawgrnhat_atm(ic,2)+pawfgrtab(iatom)%nhatfrgr(2,ic,ispden)
             pawgrnhat_atm(ic,3)=pawgrnhat_atm(ic,3)+pawfgrtab(iatom)%nhatfrgr(3,ic,ispden)
           end do
         else
           do ic=1,nfgd
             jc=2*ic-1
             pawgrnhat_atm(jc,1)=pawgrnhat_atm(jc,1)+pawfgrtab(iatom)%nhatfrgr(1,ic,ispden)
             pawgrnhat_atm(jc,2)=pawgrnhat_atm(jc,2)+pawfgrtab(iatom)%nhatfrgr(2,ic,ispden)
             pawgrnhat_atm(jc,3)=pawgrnhat_atm(jc,3)+pawfgrtab(iatom)%nhatfrgr(3,ic,ispden)
           end do
         end if
       end if
     end if

!    If needed, multiply eventually by exp(-i.q.r) phase
     if (has_phase) then
       if (cplex==1) then
         do ic=1,nfgd
           pawnhat_atm(ic)=pawnhat_atm(ic)*pawfgrtab(iatom)%expiqr(1,ic)
         end do
       else
         do ic=1,nfgd
           jc=2*ic-1
           ro_ql(1)= pawfgrtab(iatom)%expiqr(1,ic)
           ro_ql(2)=-pawfgrtab(iatom)%expiqr(2,ic)
           ro(1:2)=pawnhat_atm(jc:jc+1)
           pawnhat_atm(jc  )=ro(1)*ro_ql(1)-ro(2)*ro_ql(2)
           pawnhat_atm(jc+1)=ro(2)*ro_ql(1)+ro(1)*ro_ql(2)
         end do
       end if
       if (compute_grad) then
         if (cplex==1) then
           do ic=1,nfgd
             pawgrnhat_atm(ic,1:3)=pawgrnhat_atm(ic,1:3)*pawfgrtab(iatom)%expiqr(1,ic)
           end do
         else
           do ic=1,nfgd
             jc=2*ic-1
!            dn^hat(r)/dr_i * exp(-i.q.r)
             ro_ql(1)= pawfgrtab(iatom)%expiqr(1,ic)
             ro_ql(2)=-pawfgrtab(iatom)%expiqr(2,ic)
             do ii=1,3
               ro(1:2)=pawgrnhat_atm(jc:jc+1,ii)
               pawgrnhat_atm(jc  ,ii)=ro(1)*ro_ql(1)-ro(2)*ro_ql(2)
               pawgrnhat_atm(jc+1,ii)=ro(2)*ro_ql(1)+ro(1)*ro_ql(2)
             end do
!            -i.q_i * [n^hat(r).exp(-i.q.r)]
             ro(1:2)=pawnhat_atm(jc:jc+1)
             do ii=1,3
               pawgrnhat_atm(jc  ,ii)=pawgrnhat_atm(jc  ,ii)+qphon(ii)*ro(2)
               pawgrnhat_atm(jc+1,ii)=pawgrnhat_atm(jc+1,ii)-qphon(ii)*ro(1)
             end do
           end do
         end if
       end if
     end if

!    Add the contribution of the atom to the compensation charge
     if (compute_nhat) then
       if (cplex==1) then
         do ic=1,nfgd
           kc=pawfgrtab(iatom)%ifftsph(ic)
           pawnhat(kc,ispden)=pawnhat(kc,ispden)+pawnhat_atm(ic)
         end do
       else
         do ic=1,nfgd
           jc=2*ic-1;kc=2*pawfgrtab(iatom)%ifftsph(ic)-1
           pawnhat(kc:kc+1,ispden)=pawnhat(kc:kc+1,ispden)+pawnhat_atm(jc:jc+1)
         end do
       end if
     end if
     if (compute_grad) then
       if (cplex==1) then
         do ic=1,nfgd
           kc=pawfgrtab(iatom)%ifftsph(ic)
           pawgrnhat(kc,ispden,1:3)=pawgrnhat(kc,ispden,1:3)+pawgrnhat_atm(ic,1:3)
         end do
       else
         do ic=1,nfgd
           jc=2*ic-1;kc=2*pawfgrtab(iatom)%ifftsph(ic)-1
           do ii=1,3
             pawgrnhat(kc:kc+1,ispden,ii)=pawgrnhat(kc:kc+1,ispden,ii)+pawgrnhat_atm(jc:jc+1,ii)
           end do
         end do
       end if
     end if
!    ------------------------------------------------------------------------
!    ----- End loop over density components
!    ------------------------------------------------------------------------
   end do

   if (pawfgrtab(iatom)%gylm_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylm)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylm,(0,0))
     pawfgrtab(iatom)%gylm_allocated=0
   end if
   if (pawfgrtab(iatom)%gylmgr_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr,(0,0,0))
     pawfgrtab(iatom)%gylmgr_allocated=0
   end if
   if (pawfgrtab(iatom)%gylmgr2_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr2)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr2,(0,0,0))
     pawfgrtab(iatom)%gylmgr2_allocated=0
   end if
   if (pawfgrtab(iatom)%nhatfr_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%nhatfr)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%nhatfr,(0,0))
     pawfgrtab(iatom)%nhatfr_allocated=0
   end if
   if (pawfgrtab(iatom)%nhatfrgr_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%nhatfrgr)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%nhatfrgr,(0,0,0))
     pawfgrtab(iatom)%nhatfrgr_allocated=0
   end if
   if (pawfgrtab(iatom)%expiqr_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%expiqr)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%expiqr,(0,0))
     pawfgrtab(iatom)%expiqr_allocated=0
   end if

!  ------------------------------------------------------------------------
!  ----- End loop over atoms
!  ------------------------------------------------------------------------
 end do

!----- Free some memory
 if (compute_nhat) then
   LIBPAW_DEALLOCATE(pawnhat_atm)
 end if
 if (compute_grad) then
   LIBPAW_DEALLOCATE(pawgrnhat_atm)
 end if

!----- Reduction in case of parallelism
 if (paral_atom) then
   !call timab(48,1,tsec)
   if (compute_nhat) then
     call xpaw_mpi_sum(pawnhat,my_comm_atom,ierr)
   end if
   if (compute_grad) then
     call xpaw_mpi_sum(pawgrnhat,my_comm_atom,ierr)
   end if
   !call timab(48,2,tsec)
 end if

!----- Avoid unbalanced g-components numerical errors
! I checked and this part is not used in ABINIT, so I'll comment it out
! because it relies on the Fourier transform subroutines in ABINIT
! if (izero==1.and.compute_nhat.and.usewvl==0) then
!!  Create fake mpi_enreg to wrap fourdp
!   if (present(distribfft)) then
!     my_distribfft => distribfft
!   else
!     LIBPAW_ALLOCATE(my_distribfft,)
!     call init_distribfft_seq(my_distribfft,'f',ngfft(2),ngfft(3),'fourdp')
!   end if
!   call initmpi_seq(mpi_enreg_fft)
!   LIBPAW_DEALLOCATE(mpi_enreg_fft%distribfft)
!   if (present(comm_fft)) then
!     call set_mpi_enreg_fft(mpi_enreg_fft,comm_fft,my_distribfft,me_g0,paral_kgb)
!     my_comm_fft=comm_fft;paral_kgb_fft=paral_kgb
!   else
!     my_comm_fft=xpaw_mpi_comm_self;paral_kgb_fft=0;
!     mpi_enreg_fft%distribfft => my_distribfft
!   end if
!!  do FFT
!   LIBPAW_ALLOCATE(work,(2,nfft))
!   do ispden=1,min(2,nspden)
!     call fourdp(cplex,work,pawnhat(:,ispden),-1,mpi_enreg_fft,nfft,1,ngfft,0)
!     call zerosym(work,2,ngfft(1),ngfft(2),ngfft(3),comm_fft=my_comm_fft,distribfft=my_distribfft)
!     call fourdp(cplex,work,pawnhat(:,ispden),+1,mpi_enreg_fft,nfft,1,ngfft,0)
!   end do
!   LIBPAW_DEALLOCATE(work)
!!  Destroy fake mpi_enreg
!   call unset_mpi_enreg_fft(mpi_enreg_fft)
!   if (.not.present(distribfft)) then
!     call destroy_distribfft(my_distribfft)
!     LIBPAW_DEALLOCATE(my_distribfft)
!   end if
! end if

!----- Computation of compensation charge over real space grid
 if (compute_nhat.and.ipert==0) then
   nfftot=PRODUCT(ngfft(1:3))
   !call mean_fftr(pawnhat,tmp_compch_fft,nfft,nfftot,1,mpi_comm_sphgrid)
   ! I checked that mpi_comm_sphgrid is actually not used
   call mean_fftr(pawnhat,tmp_compch_fft,nfft,nfftot,1)
   compch_fft = tmp_compch_fft(1)
   compch_fft=compch_fft*ucvol
 end if

!Destroy atom table used for parallelism
 call free_my_atmtab(my_atmtab,my_atmtab_allocated)

 !DBG_EXIT("COLL")

end subroutine pawmknhat
!!***

!----------------------------------------------------------------------

!!****f* m_paw_nhat/pawnhatfr
!!
!! NAME
!! pawnhatfr
!!
!! FUNCTION
!! PAW: Compute frozen part of 1st-order compensation charge density nhat^(1)
!!      nhatfr(r)=Sum_ij,lm[rhoij_ij.q_ij^l.(g_l(r).Y_lm(r))^(1)]
!!      Depends on q wave vector but not on first-order wave-function.
!!
!! INPUTS
!!  ider=0: computes frozen part of compensation density
!!       1: computes frozen part of compensation density and cartesian gradients
!!  idir=direction of atomic displacement (in case of phonons perturb.)
!!  ipert=nindex of perturbation
!!  mpi_atmtab(:)=--optional-- indexes of the atoms treated by current proc
!!  comm_atom=--optional-- MPI communicator over atoms
!!  my_natom=number of atoms treated by current processor
!!  natom=total number of atoms in cell
!!  nspden=number of spin-density components
!!  ntypat=number of types of atoms
!!  pawang <type(pawang_type)>=paw angular mesh and related data
!!  pawfgrtab(my_natom) <type(pawfgrtab_type)>=atomic data given on fine rectangular grid
!!  pawrhoij(my_natom) <type(pawrhoij_type)>= Ground-State paw rhoij occupancies and related data
!!  pawtab(ntypat) <type(pawtab_type)>=paw tabulated starting data
!!  rprimd(3,3)=dimensional primitive translations for real space
!!
!! OUTPUT
!!  pawfgrtab(iatom)%nhatfr(nfgd,nspden)
!!                  frozen part of charge compensation density (inside PAW spheres)
!!                  =Sum_ij,lm[rhoij_ij.q_ij^l.(g_l(r).Y_lm(r))^(1)]
!!  === If ider==1
!!  pawfgrtab(iatom)%nhatfrgr(3,nfgd,nspden)
!!                  gradients of frozen part of charge compensation density (inside PAW spheres)
!!                  =Sum_ij,lm[rhoij_ij.q_ij^l . d/dr((g_l(r).Y_lm(r))^(1))]
!!
!! SOURCE

subroutine pawnhatfr(ider,idir,ipert,my_natom,natom,nspden,ntypat,&
&                    pawang,pawfgrtab,pawrhoij,pawtab,rprimd, &
&                    mpi_atmtab,comm_atom) ! optional arguments (parallelism)

 implicit none

!Arguments ------------------------------------
!scalars
 integer,intent(in) :: ider,idir,ipert,my_natom,natom,nspden,ntypat
 integer,optional,intent(in) :: comm_atom
 type(pawang_type),intent(in) :: pawang
!arrays
 integer,optional,target,intent(in) :: mpi_atmtab(:)
 real(dp),intent(in) :: rprimd(3,3)
 type(pawfgrtab_type),intent(inout) :: pawfgrtab(my_natom)
 type(pawrhoij_type),intent(in) :: pawrhoij(my_natom)
 type(pawtab_type),intent(in) :: pawtab(ntypat)

!Local variables-------------------------------
!scalars
 integer :: iatom,iatom_tot,ic,ils,ilslm,irhoij,isel,ispden,istr,itypat,jrhoij
 integer :: klm,klmn,lm_size,lmn2_size,lm0,lmax,lmin,mua,mub,mm,mu,my_comm_atom,nfgd,nu,optgr0,optgr1,optgr2
 logical :: my_atmtab_allocated,my_pert,paral_atom
 real(dp) :: contrib,ro
!arrays
 integer,parameter :: voigt(3,3)=reshape((/1,6,5,6,2,4,5,4,3/),(/3,3/))
 integer,parameter :: alpha(9)=(/1,2,3,3,3,2,2,1,1/),beta(9)=(/1,2,3,2,1,1,3,3,2/)
 integer,pointer :: my_atmtab(:)
 real(dp),allocatable :: nhatfr_tmp(:,:),nhatfrgr_tmp(:,:,:)

! *************************************************************************

 !DBG_ENTER("COLL")

!Only relevant for atomic displacement and strain perturbation
 if (ipert>natom.and.ipert/=natom+3.and.ipert/=natom+4) return

!Set up parallelism over atoms
 paral_atom=(present(comm_atom).and.(my_natom/=natom))
 nullify(my_atmtab);if (present(mpi_atmtab)) my_atmtab => mpi_atmtab
 my_comm_atom=xpaw_mpi_comm_self;if (present(comm_atom)) my_comm_atom=comm_atom
 call get_my_atmtab(my_comm_atom,my_atmtab,my_atmtab_allocated,paral_atom,natom,my_natom_ref=my_natom)

!Compatibility tests
 if (my_natom>0) then
   if ((pawfgrtab(1)%gylm_allocated==0.or.pawfgrtab(1)%gylmgr_allocated==0).and. &
&   pawfgrtab(1)%rfgd_allocated==0) then
     LIBPAW_BUG('pawnhatfr: pawfgrtab()%rfgd array must be allocated!')
   end if
   if (pawrhoij(1)%qphase/=1) then
     LIBPAW_BUG('pawnhatfr: not supposed to be called with qphase=2!')
   end if
 end if

 my_pert = (ipert<=natom).or.ipert==natom+3.or.ipert==natom+4

!Get correct index of strain pertubation
 if (ipert==natom+3) istr = idir
 if (ipert==natom+4) istr = idir + 3

!Loops over  atoms
 do iatom=1,my_natom
   iatom_tot=iatom;if (paral_atom) iatom_tot=my_atmtab(iatom)

!  Eventually allocate frozen nhat points
   if (my_pert) then
     if (pawfgrtab(iatom)%nhatfr_allocated==0) then
       if (allocated(pawfgrtab(iatom)%nhatfr))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%nhatfr)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%nhatfr,(pawfgrtab(iatom)%nfgd,nspden))
       pawfgrtab(iatom)%nhatfr_allocated=1
     end if
     if (ider==1.and.pawfgrtab(iatom)%nhatfrgr_allocated==0) then
       if (allocated(pawfgrtab(iatom)%nhatfrgr))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%nhatfrgr)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%nhatfrgr,(3,pawfgrtab(iatom)%nfgd,nspden))
       pawfgrtab(iatom)%nhatfrgr_allocated=1
     end if
   end if

!  Select if frozen part of nhat exists for the current perturbation
   if ((.not.my_pert).or.(pawfgrtab(iatom)%nhatfr_allocated==0)) cycle

!  Some atom-dependent quantities
   itypat=pawfgrtab(iatom)%itypat
   lm_size=pawfgrtab(iatom)%l_size**2
   lmn2_size=pawtab(itypat)%lmn2_size

!  Eventually compute g_l(r).Y_lm(r) factors for the current atom (if not already done)
   nfgd=pawfgrtab(iatom)%nfgd
   if ((pawfgrtab(iatom)%gylmgr_allocated==0).or. &
&   (pawfgrtab(iatom)%gylmgr2_allocated==0.and.ider==1)) then
     optgr0=0;optgr1=0;optgr2=0
     if(ipert==natom+3.or.ipert==natom+4)then
       if (pawfgrtab(iatom)%gylm_allocated==0) then
         if (allocated(pawfgrtab(iatom)%gylm))  then
           LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylm)
         end if
         LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylm,(nfgd,lm_size))
         pawfgrtab(iatom)%gylm_allocated=2;optgr0=1
       end if
     end if
     if (pawfgrtab(iatom)%gylmgr_allocated==0) then
       if (allocated(pawfgrtab(iatom)%gylmgr))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr,(3,nfgd,lm_size))
       pawfgrtab(iatom)%gylmgr_allocated=2;optgr1=1
     end if
     if (ider==1.and.pawfgrtab(iatom)%gylmgr2_allocated==0) then
       if (allocated(pawfgrtab(iatom)%gylmgr2))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr2)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr2,(6,nfgd,lm_size))
       pawfgrtab(iatom)%gylmgr2_allocated=2;optgr2=1
     end if
     if (optgr0+optgr1+optgr2>0) then
       call pawgylm(pawfgrtab(iatom)%gylm,pawfgrtab(iatom)%gylmgr,pawfgrtab(iatom)%gylmgr2,&
&       lm_size,nfgd,optgr0,optgr1,optgr2,pawtab(itypat),pawfgrtab(iatom)%rfgd)
     end if
   end if


!  ============ Phonons ====================================
   if (ipert<=natom) then

!    Loop over spin components
     do ispden=1,nspden

       LIBPAW_ALLOCATE(nhatfr_tmp,(3,nfgd))
       nhatfr_tmp=zero
       if (ider==1) then
         LIBPAW_ALLOCATE(nhatfrgr_tmp,(3,nfgd,3))
         nhatfrgr_tmp=zero
       end if

       jrhoij=1
       do irhoij=1,pawrhoij(iatom)%nrhoijsel
         klmn=pawrhoij(iatom)%rhoijselect(irhoij)
         klm =pawtab(itypat)%indklmn(1,klmn)
         lmin=pawtab(itypat)%indklmn(3,klmn)
         lmax=pawtab(itypat)%indklmn(4,klmn)

         if (nspden/=2) then
           ro=pawrhoij(iatom)%rhoijp(jrhoij,ispden)
         else
           if (ispden==1) then
             ro=pawrhoij(iatom)%rhoijp(jrhoij,1)+pawrhoij(iatom)%rhoijp(jrhoij,2)
           else if (ispden==2) then
             ro=pawrhoij(iatom)%rhoijp(jrhoij,1)
           end if
         end if
         ro=pawtab(itypat)%dltij(klmn)*ro

         do ils=lmin,lmax,2
           lm0=ils**2+ils+1
           do mm=-ils,ils
             ilslm=lm0+mm;isel=pawang%gntselect(lm0+mm,klm)
             if (isel>0) then
               do ic=1,nfgd
                 do mu=1,3
                   contrib=-ro*pawtab(itypat)%qijl(ilslm,klmn)&
&                   *pawfgrtab(iatom)%gylmgr(mu,ic,ilslm)
                   nhatfr_tmp(mu,ic)=nhatfr_tmp(mu,ic)+contrib
                 end do
               end do
               if (ider==1) then
                 do ic=1,nfgd
                   do nu=1,3
                     do mu=1,3
                       contrib=-ro*pawtab(itypat)%qijl(ilslm,klmn) &
&                       *pawfgrtab(iatom)%gylmgr2(voigt(mu,nu),ic,ilslm)
                       nhatfrgr_tmp(mu,ic,nu)=nhatfrgr_tmp(mu,ic,nu)+contrib
                     end do
                   end do
                 end do
               end if
             end if
           end do
         end do
         jrhoij=jrhoij+pawrhoij(iatom)%cplex_rhoij
       end do

!      Convert from cartesian to reduced coordinates
       do ic=1,nfgd
         pawfgrtab(iatom)%nhatfr(ic,ispden)= &
&         rprimd(1,idir)*nhatfr_tmp(1,ic) &
&         +rprimd(2,idir)*nhatfr_tmp(2,ic) &
&         +rprimd(3,idir)*nhatfr_tmp(3,ic)
       end do
       if (ider==1) then
         do nu=1,3
           do ic=1,nfgd
             pawfgrtab(iatom)%nhatfrgr(nu,ic,ispden)= &
&             rprimd(1,idir)*nhatfrgr_tmp(1,ic,nu) &
&             +rprimd(2,idir)*nhatfrgr_tmp(2,ic,nu) &
&             +rprimd(3,idir)*nhatfrgr_tmp(3,ic,nu)
           end do
         end do
       end if
       LIBPAW_DEALLOCATE(nhatfr_tmp)
       if (ider==1) then
         LIBPAW_DEALLOCATE(nhatfrgr_tmp)
       end if
!      End loop over spin components
     end do ! ispden


!  ============ Elastic tensor ===============================
   else if (ipert==natom+3.or.ipert==natom+4) then
!    Loop over spin components
     pawfgrtab(iatom)%nhatfr(:,:) = zero
     do ispden=1,nspden
       jrhoij=1
       do irhoij=1,pawrhoij(iatom)%nrhoijsel
         klmn=pawrhoij(iatom)%rhoijselect(irhoij)
         klm =pawtab(itypat)%indklmn(1,klmn)
         lmin=pawtab(itypat)%indklmn(3,klmn)
         lmax=pawtab(itypat)%indklmn(4,klmn)
         if (nspden/=2) then
           ro=pawrhoij(iatom)%rhoijp(jrhoij,ispden)
         else
           if (ispden==1) then
             ro=pawrhoij(iatom)%rhoijp(jrhoij,1)+pawrhoij(iatom)%rhoijp(jrhoij,2)
           else if (ispden==2) then
             ro=pawrhoij(iatom)%rhoijp(jrhoij,1)
           end if
         end if
         ro=pawtab(itypat)%dltij(klmn)*ro
         do ils=lmin,lmax,2
           lm0=ils**2+ils+1
           do mm=-ils,ils
             ilslm=lm0+mm;isel=pawang%gntselect(lm0+mm,klm)
             if (isel>0) then
!              Sum{[Q_ij_q^LM^(1)]}
               do ic=1,nfgd
                 mua=alpha(istr);mub=beta(istr)
                 pawfgrtab(iatom)%nhatfr(ic,ispden) = pawfgrtab(iatom)%nhatfr(ic,ispden)+&
&                 ro*pawtab(itypat)%qijl(ilslm,klmn)*half*(&
&                 pawfgrtab(iatom)%gylmgr(mua,ic,ilslm)*pawfgrtab(iatom)%rfgd(mub,ic)&
&                 +pawfgrtab(iatom)%gylmgr(mub,ic,ilslm)*pawfgrtab(iatom)%rfgd(mua,ic))
               end do
!              Add volume contribution
               if(istr<=3)then
                 do ic=1,nfgd
                   pawfgrtab(iatom)%nhatfr(ic,ispden) = pawfgrtab(iatom)%nhatfr(ic,ispden)+&
&                   ro*pawtab(itypat)%qijl(ilslm,klmn)*pawfgrtab(iatom)%gylm(ic,ilslm)
                 end do
               end if
               if (ider==1) then
                 LIBPAW_ERROR("nhatgr not implemented for strain perturbationxs")
!                 do ic=1,nfgd
!                   do nu=1,6
!                     do mu=1,6
!                       contrib=-ro*pawtab(itypat)%qijl(ilslm,klmn) &
!&                       *pawfgrtab(iatom)%gylmgr2(voigt(mu,nu),ic,ilslm)
!                       nhatfrgr_tmp(mu,ic,nu)=nhatfrgr_tmp(mu,ic,nu)+contrib
!                     end do
!                   end do
!                 end do
               end if
             end if
           end do
         end do
         jrhoij=jrhoij+pawrhoij(iatom)%cplex_rhoij
       end do
     end do ! ispden
   end if

!  Eventually free temporary space for g_l(r).Y_lm(r) gradients and exp(-i.q.r)
   if (pawfgrtab(iatom)%gylmgr_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr,(0,0,0))
     pawfgrtab(iatom)%gylmgr_allocated=0
   end if
   if (pawfgrtab(iatom)%gylmgr2_allocated==2) then
     LIBPAW_DEALLOCATE(pawfgrtab(iatom)%gylmgr2)
     LIBPAW_ALLOCATE(pawfgrtab(iatom)%gylmgr2,(0,0,0))
     pawfgrtab(iatom)%gylmgr2_allocated=0
   end if

!  End loop on atoms
 end do

!Destroy atom table used for parallelism
 call free_my_atmtab(my_atmtab,my_atmtab_allocated)

 !DBG_EXIT("COLL")

end subroutine pawnhatfr
!!***

!----------------------------------------------------------------------

!!****f* m_paw_nhat/nhatgrid
!! NAME
!! nhatgrid
!!
!! FUNCTION
!! Determine parts of the rectangular (fine) grid that are contained
!! inside spheres around atoms (used to compute n_hat density).
!! If corresponding option is selected, compute also g_l(r)*Y_lm(r)
!! (and derivatives) on this grid (g_l=radial shape function).
!!
!! INPUTS
!!  atindx1(natom)=index table for atoms, inverse of atindx (see gstate.f)
!!  distribfft<type(distribfft_type)>=--optional-- contains all the information related
!!                                    to the FFT parallelism and plane sharing
!!  gmet(3,3)=reciprocal space metric tensor in bohr**-2
!!  mpi_atmtab(:)=--optional-- indexes of the atoms treated by current proc
!!  comm_atom=--optional-- MPI communicator over atoms
!!  comm_fft=--optional-- MPI communicator over FFT components
!!  my_natom=number of atoms treated by current processor
!!  natom=total number of atoms in cell
!!  nattyp(ntypat)= # atoms of each type.
!!  ngfft(3)=contain all needed information about 3D FFT, see ~abinit/doc/variables/vargs.htm#ngfft
!!  ntypat=number of types of atoms in unit cell
!!  optcut= option for the cut-off radius of spheres:
!!          if optcut=0, cut-off radius=pawtab%rshp=cut-off radius of compensation charge
!!          if optcut=1, cut-off radius=pawtab%rpaw=radius of PAW augmentation regions
!!  optgr0= 1 if g_l(r)*Y_lm(r) are computed
!!  optgr1= 1 if first derivatives of g_l(r)*Y_lm(r) are computed
!!  optgr2= 1 if second derivatives of g_l(r)*Y_lm(r) are computed
!!  optrad= 1 if vectors (r-r_atom) on the fine grid around atoms have to be stored
!!  pawfgrtab(natom) <type(pawfgrtab_type)>=atomic data given on fine rectangular grid
!!  pawtab(ntypat) <type(pawtab_type)>=paw tabulated starting data
!!  rprimd(3,3)=dimensional primitive translations in real space (bohr)
!!  typat(natom)=type (integer) for each atom
!!  typord=1 if the output is ordered by type of atoms, 0 otherwise
!!  ucvol=unit cell volume in bohr**3
!!  xred(3,natom)=reduced dimensionless atomic coordinates
!!
!! OUTPUT
!!  pawfgrtab(natom)%ifftsph(nfgd)=FFT index (fine grid) of a points in paw spheres around each atom
!!  pawfgrtab(natom)%nfgd= number of (fine grid) FFT points in paw spheres around atoms
!!  if (optgr0==1)
!!    pawfgrtab(natom)%gylm(nfgd,l_size**2)= g_l(r)*Y_lm(r) around each atom
!!  if (optgr1==1)
!!    pawfgrtab(natom)%gylmgr(3,nfgd,l_size**2)= derivatives of g_l(r)*Y_lm(r) wrt cart. coordinates
!!  if (optgr2==1)
!!    pawfgrtab(natom)%gylmgr2(6,nfgd,l_size**2)= second derivatives of g_l(r)*Y_lm(r) wrt cart. coordinates
!!  if (optrad==1)
!!    pawfgrtab(natom)%rfgd(3,nfgd)= coordinates of r-r_atom around each atom
!!
!! SOURCE

!This is a wrapper for pawrfgd_fft and pawgylm from libpaw
!When transporting to another DFT software, we might replace 'distribfft'
!by directly supplying the two arrays 'fftn3_distrib' and 'ffti3_local'
!And get_my_atmtab is never called because parallelization of atoms
!is not used for PAW. See 'initmpi_atom' in src/51_manage_mpi/m_mpinfo.F90

subroutine nhatgrid(atindx1,gmet,my_natom,natom,nattyp,ngfft,ntypat,&
& optcut,optgr0,optgr1,optgr2,optrad,pawfgrtab,pawtab,rprimd,typat,ucvol,xred, &
& n3_in, fftn3_distrib, ffti3_local, mpi_atmtab, comm_atom, comm_fft, typord)
!& mpi_atmtab,comm_atom,comm_fft,distribfft,typord) ! optional arguments (parallelism)

 implicit none

!Arguments ---------------------------------------------
!scalars
 integer,intent(in) :: my_natom,natom,ntypat,optcut,optgr0,optgr1,optgr2,optrad
 integer,optional,intent(in) :: comm_atom,comm_fft,typord
 real(dp),intent(in) :: ucvol

 !Note the replacement here
 !type(distribfft_type),optional,target,intent(in)  :: distribfft
 !Note the replacement here
 integer,intent(in) :: n3_in
 integer,intent(in) :: fftn3_distrib(n3_in), ffti3_local(n3_in)
!arrays
 integer,intent(in) :: ngfft(3),typat(natom)
 integer,intent(in),target :: atindx1(natom),nattyp(ntypat)
 integer,optional,target,intent(in) :: mpi_atmtab(:)
 real(dp),intent(in) :: gmet(3,3),rprimd(3,3),xred(3,natom)
 type(pawfgrtab_type),intent(inout) :: pawfgrtab(my_natom)
 type(pawtab_type),intent(in) :: pawtab(ntypat)

!Local variables ------------------------------
!scalars
 integer :: i3,iat,iatm,iatom,iatom_,iatom_tot,itypat,lm_size,me_fft,my_comm_atom,n1,n2,n3,nfgd
 logical :: grid_found,my_atmtab_allocated,paral_atom
 real(dp) :: rcut
 character(len=500) :: msg
!arrays
 integer,allocatable :: ifftsph_tmp(:)
 integer,pointer :: my_atindx1(:),my_atmtab(:),my_nattyp(:)
 !integer, LIBPAW_CONTIGUOUS pointer :: fftn3_distrib(:),ffti3_local(:)
 real(dp) :: tsec(2)
 real(dp),allocatable :: rfgd_tmp(:,:)

! *************************************************************************

 !DBG_ENTER("COLL")

 !call timab(559,1,tsec)
 if (my_natom==0) return

!Set up parallelism over FFT
 me_fft=0
 if (present(comm_fft)) then
   me_fft=xpaw_mpi_comm_rank(comm_fft)
 end if

!Set up parallelism over atoms
!This part is not relevant, because atom parallelization
!is not used for PAW
 paral_atom=(present(comm_atom).and.(my_natom/=natom))
 nullify(my_atmtab);if (present(mpi_atmtab)) my_atmtab => mpi_atmtab
 my_comm_atom=xpaw_mpi_comm_self;if (present(comm_atom)) my_comm_atom=comm_atom
 call get_my_atmtab(my_comm_atom,my_atmtab,my_atmtab_allocated,paral_atom,natom,&
& my_natom_ref=my_natom)
 if (paral_atom) then
   LIBPAW_ALLOCATE(my_atindx1,(natom))
   LIBPAW_ALLOCATE(my_nattyp,(ntypat))
   my_atindx1(:)=0;my_nattyp(:)=0
   iat=1
   do itypat=1,ntypat
     if (my_natom>0) then
       do iatom=1,my_natom
         if(typat(my_atmtab(iatom))==itypat)then
           my_nattyp(itypat)=my_nattyp(itypat)+1
           my_atindx1(iat)=iatom
           iat=iat+1
         end if
       end do
     end if
   end do
 else
   my_atindx1 => atindx1
   my_nattyp => nattyp
 end if

!Get the distrib associated with this fft_grid

 n1=ngfft(1);n2=ngfft(2);n3=ngfft(3)
! if (present(distribfft)) then
!   grid_found=.false.
!   this part is actually not relevant
!   because a fine grid is used for PAW
!   if (n2 == distribfft%n2_coarse) then
!     if (n3== size(distribfft%tab_fftdp3_distrib)) then
!       fftn3_distrib => distribfft%tab_fftdp3_distrib
!       ffti3_local => distribfft%tab_fftdp3_local
!       grid_found=.true.
!     end if
!   end if
!   if (n2 == distribfft%n2_fine) then
!     if (n3 == size(distribfft%tab_fftdp3dg_distrib)) then
!       fftn3_distrib => distribfft%tab_fftdp3dg_distrib
!       ffti3_local => distribfft%tab_fftdp3dg_local
!       grid_found = .true.
!     end if
!   end if
!   if (.not.(grid_found)) then
!     msg='Unable to find an allocated distrib for this fft grid!'
!     LIBPAW_BUG(msg)
!   end if
! else
!   the same goes here, this part is not relevant
!   LIBPAW_ALLOCATE(fftn3_distrib,(n3))
!   LIBPAW_ALLOCATE(ffti3_local,(n3))
!   fftn3_distrib=0;ffti3_local=(/(i3,i3=1,n3)/)
! end if

!Loop over types of atom
!-------------------------------------------
 iatm=0
 do itypat=1,ntypat

   if (optcut==1) then
     rcut=pawtab(itypat)%rpaw
   else
     rcut=pawtab(itypat)%rshp
   end if

!  Loop over atoms
!  -------------------------------------------
   do iat=1,my_nattyp(itypat)
     iatm=iatm+1;iatom=my_atindx1(iatm)
     iatom_tot=iatom;if (paral_atom) iatom_tot=my_atmtab(iatom)
     iatom_=iatom;if(present(typord)) iatom_=merge(iatm,iatom,typord==1)
     lm_size=pawfgrtab(iatom_)%l_size**2

!    ------------------------------------------------------------------
!    A-Determine FFT points and r-R vectors around the atom
!    ------------------------------------------------------------------

     call pawrfgd_fft(ifftsph_tmp,gmet,n1,n2,n3,nfgd,rcut,rfgd_tmp,rprimd,ucvol,&
&     xred(:,iatom_tot),fft_distrib=fftn3_distrib,fft_index=ffti3_local,me_fft=me_fft)

!    Allocate arrays defining sphere (and related data) around current atom
     if (allocated(pawfgrtab(iatom_)%ifftsph)) then
       LIBPAW_DEALLOCATE(pawfgrtab(iatom_)%ifftsph)
     end if
     LIBPAW_ALLOCATE(pawfgrtab(iatom_)%ifftsph,(nfgd))
     pawfgrtab(iatom_)%nfgd=nfgd
     pawfgrtab(iatom_)%ifftsph(1:nfgd)=ifftsph_tmp(1:nfgd)

     if (optrad==1) then
       if (allocated(pawfgrtab(iatom_)%rfgd))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom_)%rfgd)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom_)%rfgd,(3,nfgd))
       pawfgrtab(iatom_)%rfgd_allocated=1
       pawfgrtab(iatom_)%rfgd(1:3,1:nfgd)=rfgd_tmp(1:3,1:nfgd)
     end if

     if (optgr0==1) then
       if (allocated(pawfgrtab(iatom_)%gylm))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom_)%gylm)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom_)%gylm,(nfgd,lm_size))
       pawfgrtab(iatom_)%gylm_allocated=1
     end if

     if (optgr1==1) then
       if (allocated(pawfgrtab(iatom_)%gylmgr))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom_)%gylmgr)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom_)%gylmgr,(3,nfgd,lm_size))
       pawfgrtab(iatom_)%gylmgr_allocated=1
     end if

     if (optgr2==1) then
       if (allocated(pawfgrtab(iatom_)%gylmgr2))  then
         LIBPAW_DEALLOCATE(pawfgrtab(iatom_)%gylmgr2)
       end if
       LIBPAW_ALLOCATE(pawfgrtab(iatom_)%gylmgr2,(6,nfgd,lm_size))
       pawfgrtab(iatom_)%gylmgr2_allocated=1
     end if

!    ------------------------------------------------------------------
!    B-Calculate g_l(r-R)*Y_lm(r-R) for each r around the atom R
!    ------------------------------------------------------------------
     if (optgr0+optgr1+optgr2>0) then
       call pawgylm(pawfgrtab(iatom_)%gylm,pawfgrtab(iatom_)%gylmgr,pawfgrtab(iatom_)%gylmgr2,&
&       lm_size,nfgd,optgr0,optgr1,optgr2,pawtab(itypat),rfgd_tmp(:,1:nfgd))
     end if

!    End loops over types/atoms
!    -------------------------------------------
     LIBPAW_DEALLOCATE(ifftsph_tmp)
     LIBPAW_DEALLOCATE(rfgd_tmp)
   end do
 end do

!Destroy atom tables used for parallelism
 call free_my_atmtab(my_atmtab,my_atmtab_allocated)
 if (paral_atom) then
   LIBPAW_DEALLOCATE(my_atindx1)
   LIBPAW_DEALLOCATE(my_nattyp)
 end if

 !if (.not.present(distribfft)) then
 !  LIBPAW_DEALLOCATE(fftn3_distrib)
 !  LIBPAW_DEALLOCATE(ffti3_local)
 !end if

 !call timab(559,2,tsec)

 !DBG_EXIT("COLL")

end subroutine nhatgrid
!!***

!----------------------------------------------------------------------

END MODULE m_paw_nhat
!!***

