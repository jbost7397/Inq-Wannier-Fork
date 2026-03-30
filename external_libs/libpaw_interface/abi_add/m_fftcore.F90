!!From 56_recipspace/m_kg.F90, but only the getcut subroutine
!!note that size of ngfft has been set to 3, because only nx, ny, nz are used

#include "libpaw.h"

!!****m* ABINIT/m_fftcore
!! NAME
!!  m_fftcore
!!
!! FUNCTION
!!  Low-level tools for FFT (sequential and MPI parallel version)
!!  It also provides helper functions to set up the list of G vectors
!!  inside a sphere or to count them.
!!
!! COPYRIGHT
!!  Copyright (C) 2014-2022 ABINIT group (SG, XG, AR, MG, MT)
!!  This file is distributed under the terms of the
!!  GNU General Public License, see ~abinit/COPYING
!!  or http://www.gnu.org/copyleft/gpl.txt .
!!
!! TODO
!!  1) Pass distribfft instead of MPI_enreg to simplify the API and facilitate code-reuse.
!!
!!  2) Merge this module with m_distribfft
!!
!!  3) Get rid of paral_kgb and MPI_type! This is a low-level module that may be called by other
!!     code in which paral_kgb is meaningless! FFT tables and a MPI communicator are sufficient.
!!
!! SOURCE


module m_fftcore

 USE_DEFS
 USE_MSG_HANDLING
 USE_MEMORY_PROFILING
 implicit none

 private

 public :: bound               ! Find distance**2 to boundary point of fft box nearest to kpt

 public :: switch
 public :: switchreal
 public :: scramble
 public :: fill
 public :: unfill
 public :: unswitch
 public :: unscramble
 public :: unswitchreal
 public :: mpiswitch
 public :: unmpiswitch
 
 public :: mpifft_fg2dbox
 public :: mpifft_dbox2fr
 public :: mpifft_fr2dbox
 public :: mpifft_dbox2fg

contains
!!***

!!****f* m_fftcore/bound
!! NAME
!! bound
!!
!! FUNCTION
!! For given kpt, ngfft, and gmet,
!!  Find distance**2 to boundary point of fft box nearest to kpt
!!  Find distance**2 to boundary point of fft box farthest to kpt
!!
!! INPUTS
!!  kpt(3)=real input k vector (reduced coordinates)
!!  ngfft(18)=contain all needed information about 3D FFT, see ~abinit/doc/variables/vargs.htm#ngfft
!!  gmet(3,3)=reciprocal space metric (currently in Bohr**-2)
!!
!! OUTPUT
!!  dsqmax=maximum distance**2 from k to boundary in Bohr**-2.
!!  dsqmin=minimum distance**2 from k to boundary in Bohr**-2.
!!  gbound(3)=coords of G on boundary (correspnding to gsqmin)
!!  plane=which plane min occurs in (1,2, or 3 for G1,etc).
!!
!! NOTES
!! Potential trouble: this routine was written assuming kpt lies inside
!! first Brillouin zone.  No measure is taken to fold input kpt back
!! into first zone.  Given arbitrary kpt, this will cause trouble.
!!
!! SOURCE

subroutine bound(dsqmax,dsqmin,gbound,gmet,kpt,ngfft,plane)

!Arguments ------------------------------------
!scalars
 integer,intent(out) :: plane
 real(dp),intent(out) :: dsqmax,dsqmin
!arrays
 integer,intent(in) :: ngfft(3)
 integer,intent(out) :: gbound(3)
 real(dp),intent(in) :: gmet(3,3),kpt(3)

!Local variables-------------------------------
!scalars
 integer :: i1,i1min,i2,i2min,i3,i3min
 real(dp) :: dsm,dsp
 character(len=500) :: msg

! *************************************************************************

!Set plane to impossible value
 plane=0

!look at +/- g1 planes:
 dsqmax=zero
 dsqmin=dsq(ngfft(1)/2,-ngfft(2)/2,-ngfft(3)/2,gmet,kpt)+0.01_dp
 do i2=-ngfft(2)/2,ngfft(2)/2
   do i3=-ngfft(3)/2,ngfft(3)/2
     dsp = dsq(ngfft(1)/2, i2, i3,gmet,kpt)
     dsm = dsq( - ngfft(1)/2, i2, i3,gmet,kpt)
     if (dsp>dsqmax) dsqmax = dsp
     if (dsm>dsqmax) dsqmax = dsm
     if (dsp<dsqmin) then
       dsqmin = dsp
       i1min = ngfft(1)/2
       i2min = i2
       i3min = i3
       plane=1
     end if
     if (dsm<dsqmin) then
       dsqmin = dsm
       i1min =  - ngfft(1)/2
       i2min = i2
       i3min = i3
       plane=1
     end if
   end do
 end do
!
!+/- g2 planes:
 do i1=-ngfft(1)/2,ngfft(1)/2
   do i3=-ngfft(3)/2,ngfft(3)/2
     dsp = dsq(i1,ngfft(2)/2,i3,gmet,kpt)
     dsm = dsq(i1,-ngfft(2)/2,i3,gmet,kpt)
     if (dsp>dsqmax) dsqmax = dsp
     if (dsm>dsqmax) dsqmax = dsm
     if (dsp<dsqmin) then
       dsqmin = dsp
       i1min = i1
       i2min = ngfft(2)/2
       i3min = i3
       plane=2
     end if
     if (dsm<dsqmin) then
       dsqmin = dsm
       i1min = i1
       i2min =  - ngfft(2)/2
       i3min = i3
       plane=2
     end if
   end do
 end do
!
!+/- g3 planes:
 do i1=-ngfft(1)/2,ngfft(1)/2
   do i2=-ngfft(2)/2,ngfft(2)/2
     dsp = dsq(i1,i2,ngfft(3)/2,gmet,kpt)
     dsm = dsq(i1,i2,-ngfft(3)/2,gmet,kpt)
     if (dsp>dsqmax) dsqmax = dsp
     if (dsm>dsqmax) dsqmax = dsm
     if (dsp<dsqmin) then
       dsqmin = dsp
       i1min = i1
       i2min = i2
       i3min = ngfft(3)/2
       plane=3
     end if
     if (dsm<dsqmin) then
       dsqmin = dsm
       i1min = i1
       i2min = i2
       i3min =  - ngfft(3)/2
       plane=3
     end if
   end do
 end do

 if (plane==0) then
!  Trouble: missed boundary somehow
   write(msg, '(a,a,a,3f9.4,a,3(i0,1x),a,a,a,a,a)' )&
   'Trouble finding boundary of G sphere for',ch10,&
   'kpt=',kpt(:),' and ng=',ngfft(1:3),ch10,&
   'Action : check that kpt lies',&
   'reasonably within first Brillouin zone; ',ch10,&
   'else code bug, contact ABINIT group.'
   LIBPAW_BUG(msg)
 end if

 gbound(1)=i1min
 gbound(2)=i2min
 gbound(3)=i3min

 contains

   function dsq(i1,i2,i3,gmet,kpt)

     integer :: i1,i2,i3
     real(dp) :: dsq
     real(dp) :: kpt(3),gmet(3,3)

     dsq=gmet(1,1)*(kpt(1)+dble(i1))**2&
&      +gmet(2,2)*(kpt(2)+dble(i2))**2&
&      +gmet(3,3)*(kpt(3)+dble(i3))**2&
&      +2._dp*(gmet(1,2)*(kpt(1)+dble(i1))*(kpt(2)+dble(i2))&
&      +gmet(2,3)*(kpt(2)+dble(i2))*(kpt(3)+dble(i3))&
&      +gmet(3,1)*(kpt(3)+dble(i3))*(kpt(1)+dble(i1)))
   end function dsq

end subroutine bound
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/switch
!! NAME
!!  switch
!!
!! FUNCTION
!!
!! INPUTS
!!
!! OUTPUT
!!
!! SOURCE

pure subroutine switch(n1dfft,n2,lot,n1,lzt,zt,zw)


!Arguments ------------------------------------
 integer,intent(in) :: n1dfft,n2,lot,n1,lzt
 real(dp),intent(in) :: zt(2,lzt,n1)
 real(dp),intent(inout) :: zw(2,lot,n2)

!Local variables-------------------------------
 integer :: i,j
! *************************************************************************

 do j=1,n1dfft
   do i=1,n2
     zw(1,j,i)=zt(1,i,j)
     zw(2,j,i)=zt(2,i,j)
   end do
 end do

end subroutine switch
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/switchreal
!! NAME
!!  switchreal
!!
!! FUNCTION
!!   Perform the rotation:
!!
!!     input: I2,i1,j3,(jp3)
!!     output: i1,I2,j3,(jp3)
!!
!!   and padd the signal with zeros.
!!   Used for real wavefunctions.
!!
!! INPUTS
!!  includelast
!!  n1dfft=Number of 1D FFTs to perform
!!  n2=Dimension of the transform along y
!!  n2eff
!!  lot=Cache blocking factor.
!!  n1zt
!!  lzt
!!  zt(2,lzt,n1zt)
!!
!! OUTPUT
!!  zw(2,lot,n2)
!!
!! SOURCE

pure subroutine switchreal(includelast,n1dfft,n2,n2eff,lot,n1zt,lzt,zt,zw)


!Arguments ------------------------------------
 integer,intent(in) :: includelast,n1dfft,n2,n2eff,lot,n1zt,lzt
 real(dp),intent(in) :: zt(2,lzt,n1zt)
 real(dp),intent(inout) :: zw(2,lot,n2)

!Local variables-------------------------------
 integer :: i,j
! *************************************************************************

 if (includelast==1) then

   ! Compute symmetric and antisymmetric combinations
   do j=1,n1dfft
     zw(1,j,1)=zt(1,1,2*j-1)
     zw(2,j,1)=zt(1,1,2*j  )
   end do
   do i=2,n2eff
     do j=1,n1dfft
       zw(1,j,i)=      zt(1,i,2*j-1)-zt(2,i,2*j)
       zw(2,j,i)=      zt(2,i,2*j-1)+zt(1,i,2*j)
       zw(1,j,n2+2-i)= zt(1,i,2*j-1)+zt(2,i,2*j)
       zw(2,j,n2+2-i)=-zt(2,i,2*j-1)+zt(1,i,2*j)
     end do
   end do

 else

   ! An odd number of FFTs
   ! Compute symmetric and antisymmetric combinations
   do j=1,n1dfft-1
     zw(1,j,1)=zt(1,1,2*j-1)
     zw(2,j,1)=zt(1,1,2*j  )
   end do
   zw(1,n1dfft,1)=zt(1,1,2*n1dfft-1)
   zw(2,n1dfft,1)=zero

   do i=2,n2eff
     do j=1,n1dfft-1
       zw(1,j,i)=      zt(1,i,2*j-1)-zt(2,i,2*j)
       zw(2,j,i)=      zt(2,i,2*j-1)+zt(1,i,2*j)
       zw(1,j,n2+2-i)= zt(1,i,2*j-1)+zt(2,i,2*j)
       zw(2,j,n2+2-i)=-zt(2,i,2*j-1)+zt(1,i,2*j)
     end do
     zw(1,n1dfft,i)=      zt(1,i,2*n1dfft-1)
     zw(2,n1dfft,i)=      zt(2,i,2*n1dfft-1)
     zw(1,n1dfft,n2+2-i)= zt(1,i,2*n1dfft-1)
     zw(2,n1dfft,n2+2-i)=-zt(2,i,2*n1dfft-1)
   end do
 end if

end subroutine switchreal
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/scramble
!! NAME
!!  scramble
!!
!! FUNCTION
!!  This routine performs the local rotation
!!
!!     input:  G1,R3,G2,(Gp2)
!!     output: G1,G2,R3,(Gp2)
!!
!! INPUTS
!!  i1=Index of x in the small box enclosing the G-sphere.
!!  j2
!!  lot=Cache blocking factor
!!  n1dfft=Number of 1D FFTs performed.
!!  md1,md2proc,nnd3=Used to dimension zmpi2
!!  n3=Dimension of the transform along z.
!!  zw(2,lot,n3): zw(:,1:n1dfft,n3) contains the lines transformed along z
!!
!! OUTPTU
!! zmpi2(2,md1,md2proc,nnd3)
!!
!! SOURCE

pure subroutine scramble(i1,j2,lot,n1dfft,md1,n3,md2proc,nnd3,zw,zmpi2)

!Arguments ------------------------------------
 integer,intent(in) :: i1,j2,lot,n1dfft,md1,n3,md2proc,nnd3
 real(dp),intent(in) :: zw(2,lot,n3)
 real(dp),intent(inout) :: zmpi2(2,md1,md2proc,nnd3)

!Local variables-------------------------------
!scalars
 integer :: i3,i

! *************************************************************************

 do i3=1,n3
   do i=0,n1dfft-1
     zmpi2(1,i1+i,j2,i3)=zw(1,i+1,i3)
     zmpi2(2,i1+i,j2,i3)=zw(2,i+1,i3)
   end do
 end do

end subroutine scramble
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/fill
!! NAME
!!  fill
!!
!! FUNCTION
!!   Receives a set of z-lines in reciprocal space,
!!   insert the values in the cache work array zw (no padding)
!!
!! INPUTS
!!  nd1,nd3=Dimensions of the input array zf.
!!  lot=Cache blocking factor.
!!  n1dfft=Number of 1D FFTs to perform
!!  n3=Dimension of the transform along z
!!  zf(2,nd1,nd3)=Input array
!!
!! OUTPUT
!!  zw(2,lot,n3)=Cache work array with the z-lines.
!!
!! SOURCE

pure subroutine fill(nd1,nd3,lot,n1dfft,n3,zf,zw)


!Arguments ------------------------------------
 integer,intent(in) :: nd1,nd3,lot,n1dfft,n3
 real(dp),intent(in) :: zf(2,nd1,nd3)
 real(dp),intent(inout) :: zw(2,lot,n3)

! local variables
 integer :: i1,i3

! *************************************************************************

 do i3=1,n3
   do i1=1,n1dfft
     zw(1,i1,i3)=zf(1,i1,i3)
     zw(2,i1,i3)=zf(2,i1,i3)
   end do
 end do

end subroutine fill
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/unfill
!! NAME
!!  unfill
!!
!! FUNCTION
!!  Move data from the cache work array to zf
!!
!! INPUTS
!!  nd1,nd3=Dimensions of the input array zf.
!!  lot=Cache blocking factor.
!!  n1dfft=Number of 1D FFTs to perform
!!  n3=Dimension of the transform along z
!!  zw(2,lot,n3)=Cache work array with the z-lines.
!!
!! OUTPUT
!!  zf(2,nd1,nd3)= zf(:,1:n1dfft,:1:n3) is filled with the results stored in zw
!!
!! SOURCE

pure subroutine unfill(nd1,nd3,lot,n1dfft,n3,zw,zf)


!Arguments ------------------------------------
 integer,intent(in) :: nd1,nd3,lot,n1dfft,n3
 real(dp),intent(in) :: zw(2,lot,n3)
 real(dp),intent(inout) :: zf(2,nd1,nd3)

!Local variables-------------------------------
 integer :: i1,i3
! *************************************************************************

 do i3=1,n3
   do i1=1,n1dfft
     zf(1,i1,i3)=zw(1,i1,i3)
     zf(2,i1,i3)=zw(2,i1,i3)
   end do
 end do

end subroutine unfill
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/unswitch
!! NAME
!!  unswitch
!!
!! FUNCTION
!!
!! INPUTS
!!  n1dfft=Number of 1D FFTs
!!  n2=Dimension of the transform along y
!!  lot=Cache blocking factor.
!!  n1=Dimension of the transform along x.
!!  lzt
!!  zw(2,lot,n2)=Cache work array
!!
!! OUTPUT
!!  zt(2,lzt,n1)
!!
!! SOURCE

pure subroutine unswitch(n1dfft,n2,lot,n1,lzt,zw,zt)


!Arguments ------------------------------------
 integer,intent(in) :: n1dfft,n2,lot,n1,lzt
 real(dp),intent(in) :: zw(2,lot,n2)
 real(dp),intent(inout) :: zt(2,lzt,n1)

!Local variables-------------------------------
 integer :: i,j
! *************************************************************************

 do j=1,n1dfft
   do i=1,n2
     zt(1,i,j)=zw(1,j,i)
     zt(2,i,j)=zw(2,j,i)
   end do
 end do

end subroutine unswitch
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/unscramble
!! NAME
!!  unscramble
!!
!! FUNCTION
!!
!! INPUTS
!!  i1
!!  j2
!!  lot=Cache blocking factor.
!!  n1dfft=Number of 1D FFTs to perform
!!  md1,n3,md2proc,nnd3
!!  zmpi2(2,md1,md2proc,nnd3)
!!
!! OUTPUT
!!  zw(2,lot,n3)= cache work array
!!
!! SOURCE

pure subroutine unscramble(i1,j2,lot,n1dfft,md1,n3,md2proc,nnd3,zmpi2,zw)


!Arguments ------------------------------------
 integer,intent(in) :: i1,j2,lot,n1dfft,md1,n3,md2proc,nnd3
 real(dp),intent(in) :: zmpi2(2,md1,md2proc,nnd3)
 real(dp),intent(inout) :: zw(2,lot,n3)

!Local variables-------------------------------
!scalars
 integer :: i,i3

! *************************************************************************

 do i3=1,n3
   do i=0,n1dfft-1
     zw(1,i+1,i3)=zmpi2(1,i1+i,j2,i3)
     zw(2,i+1,i3)=zmpi2(2,i1+i,j2,i3)
   end do
 end do

end subroutine unscramble
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/unswitchreal
!! NAME
!!  unswitchreal
!!
!! FUNCTION
!!
!! INPUTS
!!  n1dfft=Number of 1D FFTs to perform
!!  n2=Dimension of the transform along y
!!  n2eff=
!!  lot=Cache blocking factor.
!!  n1zt
!!  lzt
!!  zw(2,lot,n2)=Cache working array
!!
!! OUTPUT
!!  zt(2,lzt,n1)
!!
!! SOURCE

pure subroutine unswitchreal(n1dfft,n2,n2eff,lot,n1zt,lzt,zw,zt)


!Arguments ------------------------------------
 integer,intent(in) :: n1dfft,n2,n2eff,lot,n1zt,lzt
 real(dp),intent(in) :: zw(2,lot,n2)
 real(dp),intent(inout) :: zt(2,lzt,n1zt)

!Local variables-------------------------------
 integer :: i,j
! *************************************************************************

! Decompose symmetric and antisymmetric parts
 do j=1,n1dfft
   zt(1,1,2*j-1)=zw(1,j,1)
   zt(2,1,2*j-1)=zero
   zt(1,1,2*j)  =zw(2,j,1)
   zt(2,1,2*j)  =zero
 end do

 do i=2,n2eff
   do j=1,n1dfft
     zt(1,i,2*j-1)= (zw(1,j,i)+zw(1,j,n2+2-i))*half
     zt(2,i,2*j-1)= (zw(2,j,i)-zw(2,j,n2+2-i))*half
     zt(1,i,2*j)  = (zw(2,j,i)+zw(2,j,n2+2-i))*half
     zt(2,i,2*j)  =-(zw(1,j,i)-zw(1,j,n2+2-i))*half
   end do
 end do

end subroutine unswitchreal
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/mpiswitch
!! NAME
!! mpiswitch
!!
!! FUNCTION
!!
!! INPUTS
!!
!! OUTPUT
!!
!! SOURCE

pure subroutine mpiswitch(j3,n1dfft,Jp2st,J2st,lot,n1,nd2proc,nd3proc,nproc,ioption,zmpi1,zw)


!Arguments ------------------------------------
 integer,intent(in) :: j3,n1dfft,lot,n1,nd2proc,nd3proc,nproc,ioption
 integer,intent(inout) :: Jp2st,J2st
 real(dp),intent(in) :: zmpi1(2,n1,nd2proc,nd3proc,nproc)
 real(dp),intent(inout) :: zw(2,lot,n1)

!Local variables-------------------------------
 integer :: Jp2,J2,I1,ind,jj2,mfft,jjp2

! *************************************************************************
 mfft=0

 if (ioption /= 1) then
   do Jp2=Jp2st,nproc
     do J2=J2st,nd2proc
       mfft=mfft+1
       if (mfft.gt.n1dfft) then
         Jp2st=Jp2
         J2st=J2
         return
       end if
       do I1=1,n1
         zw(1,mfft,I1)=zmpi1(1,I1,J2,j3,Jp2)
         zw(2,mfft,I1)=zmpi1(2,I1,J2,j3,Jp2)
       end do
     end do
     J2st=1
   end do

 else
   do Jp2=Jp2st,nproc
     do J2=J2st,nd2proc
       mfft=mfft+1
       if (mfft.gt.n1dfft) then
         Jp2st=Jp2
         J2st=J2
         return
       end if
       ind=(Jp2-1) * nd2proc + J2
       jj2=(ind-1)/nproc +1

       !jjp2=modulo(ind,nproc) +1
       jjp2=modulo(ind-1,nproc)+1

       !in other words: mfft=(jj2-1)*nproc+jjp2 (modulo case)
       !istead of mfft=(Jjp2-1) * nd2proc + Jj2 (slice case)
       !with 1<=jjp2<=nproc, jj2=1,nd2proc
       do I1=1,n1
         ! zw(1,mfft,I1)=zmpi1(1,I1,J2,j3,Jp2)
         ! zw(2,mfft,I1)=zmpi1(2,I1,J2,j3,Jp2)
         zw(1,mfft,I1)=zmpi1(1,I1,jj2,j3,jjp2)
         zw(2,mfft,I1)=zmpi1(2,I1,jj2,j3,jjp2)
       end do
     end do
     J2st=1
   end do
 end if

end subroutine mpiswitch
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/unmpiswitch
!! NAME
!!  unmpiswitch
!!
!! FUNCTION
!!
!! INPUTS
!!
!! OUTPUT
!!
!! SOURCE

pure subroutine unmpiswitch(j3,n1dfft,Jp2st,J2st,lot,n1,nd2proc,nd3proc,nproc,ioption,zw,zmpi1)


!Arguments ------------------------------------
 integer,intent(in) :: j3,n1dfft,lot,n1,nd2proc,nd3proc,nproc,ioption
 integer,intent(inout) :: Jp2st,J2st
 real(dp),intent(in) :: zw(2,lot,n1)
 real(dp),intent(inout) :: zmpi1(2,n1,nd2proc,nd3proc,nproc)

!Local variables-------------------------------
 integer :: i1,jp2,j2,ind,jjp2,mfft,jj2

! *************************************************************************

 mfft=0
 if (ioption == 2) then
   do Jp2=Jp2st,nproc
     do J2=J2st,nd2proc
       mfft=mfft+1
       if (mfft.gt.n1dfft) then
         Jp2st=Jp2
         J2st=J2
         return
       end if
       do I1=1,n1
         zmpi1(1,I1,J2,j3,Jp2)=zw(1,mfft,I1)
         zmpi1(2,I1,J2,j3,Jp2)=zw(2,mfft,I1)
       end do
     end do
     J2st=1
   end do

 else
   do Jp2=Jp2st,nproc
     do J2=J2st,nd2proc
       mfft=mfft+1
       if (mfft.gt.n1dfft) then
         Jp2st=Jp2
         J2st=J2
         return
       end if
       ind=(Jp2-1) * nd2proc + J2
       jj2=(ind-1)/nproc +1

       !jjp2=modulo(ind,nproc) +1
       jjp2=modulo(ind-1,nproc)+1

       do I1=1,n1
         zmpi1(1,I1,jj2,j3,jjp2)=zw(1,mfft,I1)
         zmpi1(2,I1,jj2,j3,jjp2)=zw(2,mfft,I1)
       end do
     end do
     J2st=1
   end do
 end if

end subroutine unmpiswitch
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/mpifft_fg2dbox
!! NAME
!!  mpifft_fg2dbox
!!
!! FUNCTION
!!
!! INPUTS
!!
!! OUTPUT
!!
!! SOURCE

pure subroutine mpifft_fg2dbox(nfft,ndat,fofg,n1,n2,n3,n4,nd2proc,n6,fftn2_distrib,ffti2_local,me_fft,workf)


!Arguments ------------------------------------
!scalars
 integer,intent(in) :: nfft,ndat,n1,n2,n3,n4,nd2proc,n6,me_fft
!arrays
 integer,intent(in) :: fftn2_distrib(n2),ffti2_local(n2)
 real(dp),intent(in) :: fofg(2,nfft*ndat)
 real(dp),intent(inout) :: workf(2,n4,n6,nd2proc*ndat)

!Local variables-------------------------------
 integer :: idat,i1,i2,i3,i2_local,i2_ldat,fgbase

! *************************************************************************

 do idat=1,ndat
   do i3=1,n3
     do i2=1,n2
       if (fftn2_distrib(i2) == me_fft) then
         i2_local = ffti2_local(i2)
         i2_ldat = i2_local + (idat-1) * nd2proc
         fgbase= n1*(i2_local-1 + nd2proc*(i3-1)) + (idat-1) * nfft
         do i1=1,n1
           workf(1,i1,i3,i2_ldat)=fofg(1,i1+fgbase)
           workf(2,i1,i3,i2_ldat)=fofg(2,i1+fgbase)
         end do
       end if
     end do
   end do
 end do

end subroutine mpifft_fg2dbox
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/mpifft_dbox2fr
!! NAME
!!  mpifft_dbox2fr
!!
!! FUNCTION
!!
!! INPUTS
!!
!! OUTPUT
!!
!! SOURCE

pure subroutine mpifft_dbox2fr(n1,n2,n3,n4,n5,nd3proc,ndat,fftn3_distrib,ffti3_local,me_fft,workr,cplex,nfft,fofr)


!Arguments ------------------------------------
!scalars
 integer,intent(in) :: n1,n2,n3,n4,n5,nd3proc,ndat,me_fft,nfft,cplex
!!arrays
 integer,intent(in) :: fftn3_distrib(n3),ffti3_local(n3)
 real(dp),intent(in) :: workr(2,n4,n5,nd3proc*ndat)
 real(dp),intent(out) :: fofr(cplex*nfft*ndat)

!Local variables-------------------------------
 integer :: idat,i1,i2,i3,i3_local,i3_ldat,frbase

! *************************************************************************

 select case (cplex)
 case (1)

   do idat=1,ndat
     do i3=1,n3
       if( fftn3_distrib(i3) == me_fft) then
         i3_local = ffti3_local(i3)
         i3_ldat = i3_local + (idat - 1) * nd3proc
         do i2=1,n2
           frbase=n1*(i2-1+n2*(i3_local-1)) + (idat - 1) * nfft
           do i1=1,n1
             fofr(i1+frbase)=workr(1,i1,i2,i3_ldat)
           end do
         end do
       end if
     end do
   end do

 case (2)

   do idat=1,ndat
     do i3=1,n3
       if (fftn3_distrib(i3) == me_fft) then
         i3_local = ffti3_local(i3)
         i3_ldat = i3_local + (idat - 1) * nd3proc
         do i2=1,n2
           frbase=2*n1*(i2-1+n2*(i3_local-1)) + (idat - 1) * cplex * nfft
           !if (frbase > cplex*nfft*ndat - 2*n1) then
           !   write(std_out,*)i2,i3_local,frbase,cplex*nfft*ndat
           !   ABI_ERROR("frbase")
           !end if
           do i1=1,n1
             fofr(2*i1-1+frbase)=workr(1,i1,i2,i3_ldat)
             fofr(2*i1  +frbase)=workr(2,i1,i2,i3_ldat)
           end do
         end do
       end if
     end do
   end do

 case default
   !ABI_BUG("Wrong cplex")
   fofr = huge(one)
 end select

end subroutine mpifft_dbox2fr
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/mpifft_fr2dbox
!! NAME
!!  mpifft_fr2dbox
!!
!! FUNCTION
!!
!! INPUTS
!!
!! OUTPUT
!!
!! SOURCE

pure subroutine mpifft_fr2dbox(cplex,nfft,ndat,fofr,n1,n2,n3,n4,n5,nd3proc,fftn3_distrib,ffti3_local,me_fft,workr)


!Arguments ------------------------------------
!scalars
 integer,intent(in) :: cplex,nfft,ndat,n1,n2,n3,n4,n5,nd3proc,me_fft
!!arrays
 integer,intent(in) :: fftn3_distrib(n3),ffti3_local(n3)
 real(dp),intent(in) :: fofr(cplex*nfft*ndat)
 real(dp),intent(inout) :: workr(2,n4,n5,nd3proc*ndat)

!Local variables-------------------------------
 integer :: idat,i1,i2,i3,i3_local,i3_ldat,frbase

! *************************************************************************

 select case (cplex)
 case (1)

   do idat=1,ndat
     do i3=1,n3
       if( me_fft == fftn3_distrib(i3) ) then
         i3_local = ffti3_local(i3)
         i3_ldat = i3_local + (idat-1) * nd3proc
         do i2=1,n2
           frbase=n1*(i2-1+n2*(i3_local-1)) + (idat-1) * nfft
           do i1=1,n1
             workr(1,i1,i2,i3_ldat)=fofr(i1+frbase)
             workr(2,i1,i2,i3_ldat)=zero
           end do
         end do
       end if
     end do
   end do

 case (2)

   do idat=1,ndat
     do i3=1,n3
       if( me_fft == fftn3_distrib(i3) ) then
         i3_local = ffti3_local(i3)
         i3_ldat = i3_local + (idat-1) * nd3proc
         do i2=1,n2
           frbase=2*n1*(i2-1+n2*(i3_local-1)) + (idat-1) * cplex * nfft
           do i1=1,n1
             workr(1,i1,i2,i3_ldat)=fofr(2*i1-1+frbase)
             workr(2,i1,i2,i3_ldat)=fofr(2*i1  +frbase)
           end do
         end do
       end if
     end do
   end do

 case default
   !ABI_BUG("Wrong cplex")
   workr = huge(one)
 end select

end subroutine mpifft_fr2dbox
!!***

!----------------------------------------------------------------------

!!****f* m_fftcore/mpifft_dbox2fg
!! NAME
!!  mpifft_dbox2fg
!!
!! FUNCTION
!!
!! INPUTS
!!
!! OUTPUT
!!
!! SOURCE

pure subroutine mpifft_dbox2fg(n1,n2,n3,n4,nd2proc,n6,ndat,fftn2_distrib,ffti2_local,me_fft,workf,nfft,fofg)


!Arguments ------------------------------------
!scalars
 integer,intent(in) :: n1,n2,n3,n4,nd2proc,n6,ndat,me_fft,nfft
!arrays
 integer,intent(in) :: fftn2_distrib(n2),ffti2_local(n2)
 real(dp),intent(in) :: workf(2,n4,n6,nd2proc*ndat)
 real(dp),intent(out) :: fofg(2,nfft*ndat)

!Local variables-------------------------------
 integer :: idat,i1,i2,i3,i2_local,i2_ldat,fgbase
 real(dp) :: xnorm

! *************************************************************************

 xnorm=one/dble(n1*n2*n3)

 ! Transfer fft output to the original fft box
 do idat=1,ndat
   do i2=1,n2
     if( fftn2_distrib(i2) == me_fft) then
       i2_local = ffti2_local(i2)
       i2_ldat = i2_local + (idat-1) * nd2proc
       do i3=1,n3
         fgbase = n1*(i2_local - 1 + nd2proc*(i3-1)) + (idat - 1) * nfft
         do i1=1,n1
           fofg(1,i1+fgbase)=workf(1,i1,i3,i2_ldat)*xnorm
           fofg(2,i1+fgbase)=workf(2,i1,i3,i2_ldat)*xnorm
         end do
       end do
     end if
   end do
 end do

end subroutine mpifft_dbox2fg
!!***

END MODULE m_fftcore
!!***
