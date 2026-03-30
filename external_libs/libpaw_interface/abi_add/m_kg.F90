!!From 56_recipspace/m_kg.F90, but only the getcut subroutine
!!note that size of ngfft has been set to 3, because only nx, ny, nz are used

#include "libpaw.h"

!!****m* ABINIT/m_kg
!! NAME
!! m_kg
!!
!! FUNCTION
!!  Low-level functions to operate of G-vectors.
!!
!! COPYRIGHT
!!  Copyright (C) 2008-2022 ABINIT group (DCA, XG, GMR, MT, DRH, AR)
!!  This file is distributed under the terms of the
!!  GNU General Public License, see ~abinit/COPYING
!!  or http://www.gnu.org/copyleft/gpl.txt .
!!
!! SOURCE

MODULE m_kg

 USE_DEFS
 USE_MSG_HANDLING
 USE_MEMORY_PROFILING
 use m_fftcore
 
 implicit none

 public :: getcut       ! Compute cutoff G^2
 public :: getph        ! Compute three factors of one-dimensional structure factor phase

contains
!!***

!!****f* m_kg/getcut
!!
!! NAME
!! getcut
!!
!! FUNCTION
!! For input kpt, fft box dim ngfft(1:3), recip space metric gmet,
!! and kinetic energy cutoff ecut (hartree), COMPUTES:
!! if iboxcut==0:
!!   gsqcut: cut-off on G^2 for "large sphere" of radius double that
!!            of the basis sphere corresponding to ecut
!!   boxcut: where boxcut == gcut(box)/gcut(sphere).
!!                 boxcut >=2 for no aliasing.
!!                 boxcut < 1 is wrong and halts subroutine.
!! if iboxcut==1:
!!   gsqcut: cut-off on G^2 for "large sphere"
!!            containing the whole fft box
!!   boxcut: no meaning (zero)
!!
!! INPUTS
!! ecut=kinetic energy cutoff for planewave sphere (hartree)
!! gmet(3,3)=reciprocal space metric (bohr^-2)
!! iboxcut=0: compute gsqcut and boxcut with boxcut>=1
!!         1: compute gsqcut for boxcut=1 (sphere_cutoff=box_cutoff)
!! iout=unit number for output file
!! kpt(3)=input k vector (reduced coordinates--in terms of reciprocal lattice primitive translations)
!! ngfft(18)=contain all needed information about 3D FFT, see ~abinit/doc/variables/vargs.htm#ngfft
!!
!! OUTPUT
!! boxcut=defined above (dimensionless), ratio of basis sphere
!!  diameter to fft box length (smallest value)
!! gsqcut=Fourier cutoff on G^2 for "large sphere" of radius double
!!  that of the basis sphere--appropriate for charge density rho(G),
!!  Hartree potential, and pseudopotentials
!!
!! NOTES
!! 2*gcut arises from rho(g)=sum g prime (psi(g primt)*psi(g prime+g))
!!               where psi(g) is only nonzero for |g| <= gcut).
!! ecut (currently in hartree) is proportional to gcut(sphere)**2.
!!
!! SOURCE

subroutine getcut(ecut,gmet,gsqcut,iboxcut,ngfft)

!Arguments ------------------------------------
!scalars
 integer,intent(in) :: iboxcut
 real(dp),intent(in) :: ecut
 real(dp),intent(out) :: gsqcut
!arrays
 integer,intent(in) :: ngfft(3)
 real(dp),intent(in) :: gmet(3,3)

!Local variables-------------------------------
!scalars
 integer :: plane
 real(dp) :: boxsq,cutrad,ecut_pw,effcut,largesq,sphsq
 character(len=1000) :: msg
!arrays
 integer :: gbound(3)
 real(dp) :: boxcut, kpt(3) = 0._dp

! *************************************************************************

 ! This is to treat the case in which ecut has not been initialized e.g. for wavelet computations.
 ! The default for ecut is -1.0 , allowed only for wavelets calculations
 ecut_pw=ecut
 if(ecut<-tol8)ecut_pw=ten

 !gcut(box)**2=boxsq; gcut(sphere)**2=sphsq
 !get min. d**2 to boundary of fft box:
 !(gmet sets dimensions: bohr**-2)
 !ecut(sphere)=0.5*(2 pi)**2 * sphsq:
 call bound(largesq,boxsq,gbound,gmet,kpt,ngfft,plane)
 effcut=0.5_dp * (two_pi)**2 * boxsq
 sphsq=2._dp*ecut_pw/two_pi**2

 if (iboxcut/=0) then
   boxcut=10._dp
   gsqcut=(largesq/sphsq)*(2.0_dp*ecut)/two_pi**2

   write(msg, '(a,a,3f8.4,a,3i4,a,a,f11.3,a,a)' ) ch10,&
   ' getcut: wavevector=',kpt,'  ngfft=',ngfft(1:3),ch10,&
   '         ecut(hartree)=',ecut_pw+tol8,ch10,'=> whole FFT box selected'
   call wrtout(std_out,msg)
 else

  ! Get G^2 cutoff for sphere of double radius of basis sphere
  ! for selecting G s for rho(G), V_Hartree(G), and V_psp(G)--
  ! cut off at fft box boundary or double basis sphere radius, whichever
  ! is smaller.  If boxcut were 2, then relation would be
  ! $ecut_eff = (1/2) * (2 Pi Gsmall)^2 and gsqcut=4*Gsmall^2$.
   boxcut = sqrt(boxsq/sphsq)
   cutrad = min(2.0_dp,boxcut)
   gsqcut = (cutrad**2)*(2.0_dp*ecut_pw)/two_pi**2

   if(ecut>-tol8)then

     write(msg, '(a,a,3f8.4,a,3i4,a,a,f11.3,3x,a,f10.5)' ) ch10,&
     ' getcut: wavevector=',kpt,'  ngfft=',ngfft(1:3),ch10,&
     '         ecut(hartree)=',ecut+tol8,'=> boxcut(ratio)=',boxcut+tol8
     call wrtout(std_out,msg)

     if (boxcut<1.0_dp) then
       write(msg, '(9a,f12.6,6a)' )&
       'Choice of acell, ngfft, and ecut',ch10,&
       '===> basis sphere extends BEYOND fft box !',ch10,&
       'Recall that boxcut=Gcut(box)/Gcut(sphere)  must be > 1.',ch10,&
       'Action: try larger ngfft or smaller ecut.',ch10,&
       'Note that ecut=effcut/boxcut**2 and effcut=',effcut+tol8,ch10,&
       'This situation might happen when optimizing the cell parameters.',ch10,&
       'Your starting geometry might be crazy.',ch10,&
       'See https://wiki.abinit.org/doku.php?id=howto:troubleshooting#incorrect_initial_geometry .'
       LIBPAW_ERROR(msg)
     end if

     if (boxcut>2.2_dp) then
       write(msg, '(a,a,a,a,a,a,a,a,a,a,a,f12.6,a,a)' ) ch10,&
       ' getcut : COMMENT -',ch10,&
       '  Note that boxcut > 2.2 ; recall that',' boxcut=Gcut(box)/Gcut(sphere) = 2',ch10,&
       '  is sufficient for exact treatment of convolution.',ch10,&
       '  Such a large boxcut is a waste : you could raise ecut',ch10,&
       '  e.g. ecut=',effcut*0.25_dp+tol8,' Hartrees makes boxcut=2',ch10
       call wrtout(std_out,msg)
     end if

     if (boxcut<1.5_dp) then
       write(msg, '(15a)' ) ch10,&
       ' getcut : WARNING -',ch10,&
       '  Note that boxcut < 1.5; this usually means',ch10,&
       '  that the forces are being fairly strongly affected by',' the smallness of the fft box.',ch10,&
       '  Be sure to test with larger ngfft(1:3) values.',ch10,&
       '  This situation might happen when optimizing the cell parameters.',ch10,&
       '  Your starting geometry might be crazy.',ch10,&
       '  See https://wiki.abinit.org/doku.php?id=howto:troubleshooting#incorrect_initial_geometry .'
       call wrtout(std_out,msg)
     end if

   end if

 end if  ! iboxcut

end subroutine getcut
!!***

!!****f* m_kg/getph
!!
!! NAME
!! getph
!!
!! FUNCTION
!! Compute three factors of one-dimensional structure factor phase
!! for input atomic coordinates, for all planewaves which fit in fft box.
!! The storage of these atomic factors is made according to the
!! values provided by the index table atindx. This will save time in nonlop.
!!
!! INPUTS
!!  atindx(natom)=index table for atoms (see gstate.f)
!!  natom=number of atoms in cell.
!!  n1,n2,n3=dimensions of fft box (ngfft(3)).
!!  xred(3,natom)=reduced atomic coordinates.
!!
!! OUTPUT
!!  ph1d(2,(2*n1+1)*natom+(2*n2+1)*natom+(2*n3+1)*natom)=exp(2Pi i G.xred) for
!!   integer vector G with components ranging from -nj <= G <= nj.
!!   Real and imag given in usual Fortran convention.
!!
!! SOURCE

subroutine getph(atindx,natom,n1,n2,n3,ph1d,xred)

!Arguments ------------------------------------
!scalars
 integer,intent(in) :: n1,n2,n3,natom
!arrays
 integer,intent(in) :: atindx(natom)
 real(dp),intent(in) :: xred(3,natom)
 real(dp),intent(out) :: ph1d(:,:)

!Local variables-------------------------------
!scalars
 integer,parameter :: im=2,re=1
 integer :: i1,i2,i3,ia,ii,ph1d_size1,ph1d_size2,ph1d_sizemin
 !character(len=500) :: msg
 real(dp) :: arg

! *************************************************************************

 ph1d_size1=size(ph1d,1);ph1d_size2=size(ph1d,2)
 ph1d_sizemin=(2*n1+1+2*n2+1+2*n3+1)*natom
 if (ph1d_size1/=2.or.ph1d_size2<ph1d_sizemin) then
   LIBPAW_BUG('Wrong ph1d sizes!')
 end if

 do ia=1,natom

!  Store the phase factor of atom number ia in place atindx(ia)
   i1=(atindx(ia)-1)*(2*n1+1)
   i2=(atindx(ia)-1)*(2*n2+1)+natom*(2*n1+1)
   i3=(atindx(ia)-1)*(2*n3+1)+natom*(2*n1+1+2*n2+1)

   do ii=1,2*n1+1
     arg=two_pi*dble(ii-1-n1)*xred(1,ia)
     ph1d(re,ii+i1)=dcos(arg)
     ph1d(im,ii+i1)=dsin(arg)
   end do

   do ii=1,2*n2+1
     arg=two_pi*dble(ii-1-n2)*xred(2,ia)
     ph1d(re,ii+i2)=dcos(arg)
     ph1d(im,ii+i2)=dsin(arg)
   end do

   do ii=1,2*n3+1
     arg=two_pi*dble(ii-1-n3)*xred(3,ia)
     ph1d(re,ii+i3)=dcos(arg)
     ph1d(im,ii+i3)=dsin(arg)
   end do

 end do

!This is to avoid uninitialized ph1d values
 if (ph1d_sizemin<ph1d_size2) then
   ph1d(:,ph1d_sizemin+1:ph1d_size2)=zero
 end if

end subroutine getph
!!***

end module m_kg
!!***
