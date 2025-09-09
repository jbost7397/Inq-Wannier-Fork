!!****m* m_mkrho/m_mkrho
!! NAME
!!  m_mkrho
!!
!! FUNCTION
!!
!!
!! COPYRIGHT
!!  Copyright (C) 1998-2022 ABINIT group (DCA, XG, GMR, LSI, AR, MB, MT)
!!  This file is distributed under the terms of the
!!  GNU General Public License, see ~abinit/COPYING
!!  or http://www.gnu.org/copyleft/gpl.txt .
!!
!! SOURCE

#include "libpaw.h"

module m_mkrho

 USE_DEFS
 USE_MSG_HANDLING
 USE_MPI_WRAPPERS
 USE_MEMORY_PROFILING

 use m_fft,          only : zerosym
 use m_pawtab,       only : pawtab_type
 use m_sg2002

 implicit none

 private
!!***

 public :: initro

 private :: atom_length
!!***

contains
!!***

!!****f* m_atomdata/atom_length
!! NAME
!! atom_length
!!
!! FUNCTION
!! Return atomic decay length for one given type of atom.
!! This length is used to generate an approximate atomic gaussian density
!! in reciprocal space:   n^AT(G)=exp[-(2pi.length.G)^2]
!!
!! INPUTS
!! densty=parameter for initialisation of the density of this atom type
!!        if densty>0, returned decay length if densty !
!! zion=charge on current type of atom (real number)
!! znucl=atomic number, for current type of atom
!!
!! OUTPUT
!! length=decay lenth
!!
!! SOURCE

function atom_length(densty,zion,znucl) result(length)

!Arguments ------------------------------------
!scalars
 real(dp),intent(in) :: densty,zion,znucl
 real(dp) :: length

!Local variables-------------------------------
!scalars
 integer :: nval
 real(dp) :: coreel
!arrays
 real(dp) :: data_length(16)

! *************************************************************************

!Either use the input value, or the default value, tabulated now.
 if(abs(densty)>tol10)then
   length=densty
 else

!  Count the number of core electrons.
   coreel=znucl-zion
!  Round the number of valence electrons
   nval=nint(zion)

!  For each set of core electron numbers, there are different decay lengths,
!  they start from nval=1, and proceed by group of 5, until a default is used

   if (nval==0) then
     length=zero

!    Bare ions : adjusted on 1h and 2he only
   else if(coreel<0.5)then
     data_length(1:4)=(/ .6_dp,.4_dp,.3_dp,.25_dp /)
     length=.2_dp
     if(nval<=4)length=data_length(nval)

!    1s2 core : adjusted on 3li, 6c, 7n, and 8o
   else if(coreel<2.5)then
     data_length(1:8)=(/ 1.8_dp,1.4_dp,1.0_dp ,.7_dp,.6_dp,&
&     .5_dp, .4_dp, .35_dp /)
     length=.3_dp
     if(nval<=8)length=data_length(nval)

!    Ne core (1s2 2s2 2p6) : adjusted on 11na, 13al, 14si and 17cl
   else if(coreel<10.5)then
     data_length(1:10)=(/ 2.0_dp,1.6_dp,1.25_dp,1.1_dp,1.0_dp,&
&     .9_dp, .8_dp, .7_dp , .7_dp, .7_dp  /)
     length=.6_dp
     if(nval<=10)length=data_length(nval)

!    Mg core (1s2 2s2 2p6 3s2) : adjusted on 19k, and on coreel==10
   else if(coreel<12.5)then
     data_length(1:10)=(/ 1.9_dp,1.5_dp,1.15_dp,1.0_dp,0.9_dp,&
&     .8_dp, .7_dp, .6_dp , .6_dp, .6_dp  /)
     length=.5_dp
     if(nval<=10)length=data_length(nval)

!    Ar core (Ne + 3s2 3p6) : adjusted on 20ca, 25mn and 30zn
   else if(coreel<18.5)then
     data_length(1:12)=(/ 2.0_dp ,1.8_dp ,1.5_dp,1.2_dp ,1.0_dp,&
&     .9_dp , .85_dp, .8_dp, .75_dp, .7_dp,&
&     .65_dp, .65_dp /)
     length=.6_dp
     if(nval<=12)length=data_length(nval)

!    Full 3rd shell core (Ar + 3d10) : adjusted on 31ga, 34se and 38sr
   else if(coreel<28.5)then
     data_length(1:14)=(/ 1.5_dp ,1.25_dp,1.15_dp,1.05_dp,1.00_dp,&
&     .95_dp, .95_dp, .9_dp , .9_dp , .85_dp,&
&     .85_dp, .80_dp, .8_dp , .75_dp         /)
     length=.7_dp
     if(nval<=14)length=data_length(nval)

!    Krypton core (Ar + 3d10 4s2 4p6) : adjusted on 39y, 42mo and 48cd
   else if(coreel<36.5)then
     data_length(1:12)=(/ 2.0_dp ,2.00_dp,1.60_dp,1.40_dp,1.25_dp,&
&     1.10_dp,1.00_dp, .95_dp, .90_dp, .85_dp,&
&     .80_dp, .75_dp /)
     length=.7_dp
     if(nval<=12)length=data_length(nval)

!    For the remaining elements, consider a function of nval only
   else
     data_length(1:12)=(/ 2.0_dp ,2.00_dp,1.55_dp,1.25_dp,1.15_dp,&
&     1.10_dp,1.05_dp,1.0_dp , .95_dp , .9_dp,&
&     .85_dp, .85_dp /)
     length=.8_dp
     if(nval<=12)length=data_length(nval)

   end if

 end if  ! End the choice between default and no-default

!DEBUG
!Here, use the previous default
!length=1.2_dp
!ENDDEBUG

end function atom_length

!!****f* m_mkrho/initro
!! NAME
!! initro
!!
!! FUNCTION
!! Initialize the density using either:
!!  - a gaussian of adjustable decay length (norm-conserving psp)
!!  - PS atomic valence density from psp file (PAW or NC psps with valence change in the pp file)
!!
!! INPUTS
!! atindx(natom)=index table for atoms (see gstate.f)
!! densty(ntypat,4)=parameters for initialisation of the density of each atom type
!! gmet(3,3)=reciprocal space metric (Bohr**-2)
!! gsqcut=cutoff G**2 for included G s in fft box (larger sphere).
!! izero=if 1, unbalanced components of rho(g) have to be set to zero
!! mgfft=maximum size of 1D FFTs
!! mpi_enreg=information about mpi parallelization
!! mqgrid=number of grid pts in q array for n^AT(q) spline.
!! natom=number of atoms in cell.
!! nattyp(ntypat)=number of atoms of each type in cell.
!! nfft=(effective) number of FFT grid points (for this processor)
!! ngfft(18)=contain all needed information about 3D FFT, see ~abinit/doc/variables/vargs.htm#ngfft
!! ntypat=number of types of atoms in cell.
!! nspden=number of spin-density components
!! pawtab(ntypat*usepaw) <type(pawtab_type)>=paw tabulated starting data
!! ph1d(2,3*(2*mgfft+1)*natom)=1-dim phase information for given atom coordinates.
!! qgrid(mqgrid)=q grid for spline atomic valence density n^AT(q) from 0 to qmax.
!! spinat(3,natom)=initial spin of each atom, in unit of hbar/2.
!! ucvol=unit cell volume (Bohr**3).
!! usepaw= 0 for non paw calculation; =1 for paw calculation
!! zion(ntypat)=charge on each type of atom (real number)
!! znucl(ntypat)=atomic number, for each type of atom
!!
!! OUTPUT
!! rhog(2,nfft)=initialized total density in reciprocal space
!! rhor(nfft,nspden)=initialized total density in real space.
!!         as well as spin-up part if spin-polarized
!!
!! SOURCE

subroutine initro(pawtab,atindx,densty,gmet,gsqcut,izero,mgfft,mqgrid,natom,nattyp,&
&  nfft,ngfft,nspden,ntypat,ph1d,qgrid,rhor,spinat,ucvol,usepaw,zion,znucl,&
&  n2_in,fftn2_distrib,ffti2_local,n3_in,fftn3_distrib,ffti3_local)


!Arguments ------------------------------------
!scalars
 integer,intent(in) :: izero,mgfft,mqgrid,natom,nfft,nspden,ntypat
 integer,intent(in) :: usepaw
 real(dp),intent(in) :: gsqcut,ucvol
 !type(mpi_type),intent(in) :: mpi_enreg
!arrays
 integer,intent(in) :: atindx(natom),nattyp(ntypat),ngfft(3)
 real(dp),intent(in) :: densty(ntypat,4),gmet(3,3),ph1d(2,3*(2*mgfft+1)*natom)
 real(dp),intent(in) :: qgrid(mqgrid),spinat(3,natom),zion(ntypat)
 real(dp),intent(in) :: znucl(ntypat)
 real(dp),intent(out) :: rhor(nfft,nspden)
 
 type(pawtab_type),intent(in) :: pawtab(ntypat*usepaw)

!Local variables-------------------------------
!The decay lengths should be optimized element by element, and even pseudopotential by pseudopotential.
!scalars
 integer,parameter :: im=2,re=1
 integer :: i1,i2,i3,ia,ia1,ia2,id1,id2,id3,ig1,ig2,ig3,ii,ispden
 integer :: itypat,jj,jtemp,me_fft,n1,n2,n3
 real(dp),parameter :: tolfix=1.000000001_dp
 real(dp) :: aa,alf2pi2,bb,cc,cutoff,dd,diff,dq,dq2div6,dqm1,fact,fact0,gmag
 real(dp) :: gsquar,rhoat,sfi,sfr
 real(dp) :: xnorm
 character(len=500) :: message
!arrays
 integer :: n2_in,n3_in
 integer, intent(in) :: fftn2_distrib(n2_in),ffti2_local(n2_in),fftn3_distrib(n3_in),ffti3_local(n3_in)
 real(dp),allocatable :: length(:),spinat_indx(:,:),work(:),rhog(:,:)
 logical,allocatable :: use_gaussian(:)

! *************************************************************************

 if(nspden==4)then
   write(std_out,*)' initro : might work yet for nspden=4 (not checked)'
   write(std_out,*)' spinat',spinat(1:3,1:natom)
!  stop
 end if

 n1=ngfft(1)
 n2=ngfft(2)
 n3=ngfft(3)
 me_fft = xpaw_mpi_comm_rank(xpaw_mpi_world)
 LIBPAW_ALLOCATE(work,(nfft))
 LIBPAW_ALLOCATE(spinat_indx,(3,natom))
 LIBPAW_ALLOCATE(rhog,(2,nfft))

!Transfer the spinat array to an array in which the atoms have the proper order, type by type.
 do ia=1,natom
   spinat_indx(:,atindx(ia))=spinat(:,ia)
 end do

!Check whether the values of spinat are acceptable
 if(nspden==2)then
   ia1=1
   do itypat=1,ntypat
!    ia1,ia2 sets range of loop over atoms:
     ia2=ia1+nattyp(itypat)-1
     do ia=ia1,ia2
       if( sqrt(spinat_indx(1,ia)**2+spinat_indx(2,ia)**2+spinat_indx(3,ia)**2) &
&       > abs(zion(itypat))*(1.0_dp + epsilon(0.0_dp)) ) then
         write(message, '(a,a,a,a,i4,a,a,3es11.4,a,a,a,es11.4)' ) ch10,&
&         ' initro : WARNING - ',ch10,&
&         '  For type-ordered atom number ',ia,ch10,&
&         '  input spinat=',spinat_indx(:,ia),'  is larger, in magnitude,',ch10,&
&         '  than zion(ia)=',zion(itypat)
         call wrtout(std_out,message,'COLL')
         call wrtout(ab_out,message,'COLL')
       end if
     end do
     ia1=ia2+1
   end do
 end if

!Compute the decay length of each type of atom
 LIBPAW_ALLOCATE(length,(ntypat))
 LIBPAW_ALLOCATE(use_gaussian,(ntypat))
 jtemp=0
 do itypat=1,ntypat

   use_gaussian(itypat)=.true.
   use_gaussian(itypat)=(pawtab(itypat)%has_tvale==0)
   if (.not.use_gaussian(itypat)) jtemp=jtemp+1

   if (use_gaussian(itypat)) then
     length(itypat) = atom_length(densty(itypat,1),zion(itypat),znucl(itypat))
     write(message,'(a,i3,a,f12.4,a,a,a,f12.4,a,i3,a,es12.4,a)' )&
&     ' initro: for itypat=',itypat,', take decay length=',length(itypat),',',ch10,&
&     ' initro: indeed, coreel=',znucl(itypat)-zion(itypat),', nval=',int(zion(itypat)),' and densty=',densty(itypat,1),'.'
     call wrtout(std_out,message,'COLL')
   else
     write(message,"(a,i3,a)")' initro: for itypat=',itypat,", take pseudo charge density from pp file"
     call wrtout(std_out,message,"COLL")
   end if

 end do

 if (jtemp>0) then
   dq=(qgrid(mqgrid)-qgrid(1))/dble(mqgrid-1)
   dqm1=1.0_dp/dq
   dq2div6=dq**2/6.0_dp
 end if

 cutoff=gsqcut*tolfix
 xnorm=1.0_dp/ucvol

 id1=n1/2+2
 id2=n2/2+2
 id3=n3/2+2

 if(nspden /= 4) then

   do ispden=nspden,1,-1
!    This loop overs spins will actually be as follows :
!    ispden=2 for spin up
!    ispden=1 for total spin (also valid for non-spin-polarized calculations)
!    The reverse ispden order is chosen, in order to end up with
!    rhog containing the proper total density.

     rhog(:,:)=zero

     ia1=1
     do itypat=1,ntypat

       if (use_gaussian(itypat)) alf2pi2=(two_pi*length(itypat))**2

!      ia1,ia2 sets range of loop over atoms:
       ia2=ia1+nattyp(itypat)-1
       ii=0
       jtemp=0

       do i3=1,n3
         ig3=i3-(i3/id3)*n3-1
         do i2=1,n2
           ig2=i2-(i2/id2)*n2-1
           if (fftn2_distrib(i2)==me_fft) then
             do i1=1,n1

               ig1=i1-(i1/id1)*n1-1
               ii=ii+1
!              gsquar=gsq_ini(ig1,ig2,ig3)
               gsquar=dble(ig1*ig1)*gmet(1,1)+dble(ig2*ig2)*gmet(2,2)+&
&               dble(ig3*ig3)*gmet(3,3)+dble(2*ig1*ig2)*gmet(1,2)+&
&               dble(2*ig2*ig3)*gmet(2,3)+dble(2*ig3*ig1)*gmet(3,1)

!              Skip G**2 outside cutoff:
               if (gsquar<=cutoff) then

!                Assemble structure factor over all atoms of given type,
!                also taking into account the spin-charge on each atom:
                 sfr=zero;sfi=zero
                 if(ispden==1)then
                   do ia=ia1,ia2
                     sfr=sfr+phre_ini(ig1,ig2,ig3,ia)
                     sfi=sfi-phimag_ini(ig1,ig2,ig3,ia)
                   end do
                   if (use_gaussian(itypat)) then
                     sfr=sfr*zion(itypat)
                     sfi=sfi*zion(itypat)
                   end if
                 else
                   fact0=half;if (.not.use_gaussian(itypat)) fact0=half/zion(itypat)
                   do ia=ia1,ia2
!                    Here, take care only of the z component
                     fact=fact0*(zion(itypat)+spinat_indx(3,ia))
                     sfr=sfr+phre_ini(ig1,ig2,ig3,ia)*fact
                     sfi=sfi-phimag_ini(ig1,ig2,ig3,ia)*fact
                   end do
                 end if

!                Charge density integrating to one
                 if (use_gaussian(itypat)) then
                   rhoat=xnorm*exp(-gsquar*alf2pi2)
!                  Multiply structure factor times rhoat (atomic density in reciprocal space)
                   rhog(re,ii)=rhog(re,ii)+sfr*rhoat
                   rhog(im,ii)=rhog(im,ii)+sfi*rhoat
                 else
                   gmag=sqrt(gsquar)
                   jj=1+int(gmag*dqm1)
                   diff=gmag-qgrid(jj)
                   bb = diff*dqm1
                   aa = one-bb
                   cc = aa*(aa**2-one)*dq2div6
                   dd = bb*(bb**2-one)*dq2div6
                   if (usepaw == 1) then
                     rhoat=(aa*pawtab(itypat)%tvalespl(jj,1)+bb*pawtab(itypat)%tvalespl(jj+1,1)+&
&                     cc*pawtab(itypat)%tvalespl(jj,2)+dd*pawtab(itypat)%tvalespl(jj+1,2)) *xnorm
                   else
                     ABI_BUG('Initialization of density is non consistent.')
                   end if
!                  Multiply structure factor times rhoat (atomic density in reciprocal space)
                   rhog(re,ii)=rhog(re,ii)+sfr*rhoat
                   rhog(im,ii)=rhog(im,ii)+sfi*rhoat
                 end if

               else
                 jtemp=jtemp+1
               end if

             end do ! End loop on i1
           end if
         end do ! End loop on i2
       end do ! End loop on i3
       ia1=ia2+1

     end do ! End loop on type of atoms

!    Set contribution of unbalanced components to zero
     if (izero==1) then
       call zerosym(rhog,2,n1,n2,n3,n2_in,fftn2_distrib,ffti2_local)
     end if
     !write(std_out,*)"initro: ispden, ucvol * rhog(:2,1)",ispden, ucvol * rhog(:2,1)

!    Note, we end with ispden=1, so that rhog contains the total density
     call sg2002_mpifourdp(1,nfft,ngfft,1,1,& !cplx = 1; ndat = 1;isign = 1, means G -> r
       fftn2_distrib,ffti2_local,fftn3_distrib,ffti3_local,rhog,work,xpaw_mpi_world)
     !call fourdp(1,rhog,work,1,mpi_enreg,nfft,1,ngfft,0)
     rhor(:,ispden)=work(:)
   end do ! End loop on spins

 else if(nspden==4) then
   do ispden=nspden,1,-1
!    This loop overs spins will actually be as follows :
!    ispden=2,3,4 for mx,my,mz
!    ispden=1 for total spin (also valid for non-spin-polarized calculations)
!    The reverse ispden order is chosen, in order to end up with
!    rhog containing the proper total density.

     rhog(:,:)=zero

     ia1=1
     do itypat=1,ntypat

       if (use_gaussian(itypat)) alf2pi2=(two_pi*length(itypat))**2

!      ia1,ia2 sets range of loop over atoms:
       ia2=ia1+nattyp(itypat)-1
       ii=0
       jtemp=0
       do i3=1,n3
         ig3=i3-(i3/id3)*n3-1
         do i2=1,n2
           ig2=i2-(i2/id2)*n2-1
           if (fftn2_distrib(i2)==me_fft) then
             do i1=1,n1

               ig1=i1-(i1/id1)*n1-1
               ii=ii+1
!              gsquar=gsq_ini(ig1,ig2,ig3)
               gsquar=dble(ig1*ig1)*gmet(1,1)+dble(ig2*ig2)*gmet(2,2)+&
&               dble(ig3*ig3)*gmet(3,3)+dble(2*ig1*ig2)*gmet(1,2)+&
&               dble(2*ig2*ig3)*gmet(2,3)+dble(2*ig3*ig1)*gmet(3,1)

!              Skip G**2 outside cutoff:
               if (gsquar<=cutoff) then

!                Assemble structure factor over all atoms of given type,
!                also taking into account the spin-charge on each atom:
                 sfr=zero;sfi=zero
                 if(ispden==1)then
                   do ia=ia1,ia2
                     sfr=sfr+phre_ini(ig1,ig2,ig3,ia)
                     sfi=sfi-phimag_ini(ig1,ig2,ig3,ia)
                   end do
                   if (use_gaussian(itypat)) then
                     sfr=sfr*zion(itypat)
                     sfi=sfi*zion(itypat)
                   end if
                 else
                   fact0=one;if (.not.use_gaussian(itypat)) fact0=one/zion(itypat)
                   do ia=ia1,ia2
!                    Here, take care of the components of m
                     fact=fact0*spinat_indx(ispden-1,ia)
                     sfr=sfr+phre_ini(ig1,ig2,ig3,ia)*fact
                     sfi=sfi-phimag_ini(ig1,ig2,ig3,ia)*fact
                   end do
                 end if

!                Charge density integrating to one
                 if (use_gaussian(itypat)) then
                   rhoat=xnorm*exp(-gsquar*alf2pi2)
                 else
                   gmag=sqrt(gsquar)
                   jj=1+int(gmag*dqm1)
                   diff=gmag-qgrid(jj)
                   bb = diff*dqm1
                   aa = one-bb
                   cc = aa*(aa**2-one)*dq2div6
                   dd = bb*(bb**2-one)*dq2div6
                   if (usepaw == 1) then
                     rhoat=(aa*pawtab(itypat)%tvalespl(jj,1)+bb*pawtab(itypat)%tvalespl(jj+1,1)+&
&                     cc*pawtab(itypat)%tvalespl(jj,2)+dd*pawtab(itypat)%tvalespl(jj+1,2)) *xnorm
                   else
                     ABI_BUG('Initialization of density is non consistent.')
                   end if
                 end if

!                Multiply structure factor times rhoat (atomic density in reciprocal space)
                 rhog(re,ii)=rhog(re,ii)+sfr*rhoat
                 rhog(im,ii)=rhog(im,ii)+sfi*rhoat
               else
                 jtemp=jtemp+1
               end if

             end do ! End loop on i1
           end if
         end do ! End loop on i2
       end do ! End loop on i3
       ia1=ia2+1
     end do ! End loop on type of atoms

!    Set contribution of unbalanced components to zero
     if (izero==1) then
       call zerosym(rhog,2,n1,n2,n3,n2_in,fftn2_distrib,ffti2_local)
     end if
     !write(std_out,*)"initro: ispden, ucvol * rhog(:2,1)",ispden, ucvol * rhog(:2,1)

!    Note, we end with ispden=1, so that rhog contains the total density
     !call fourdp(1,rhog,work,1,mpi_enreg,nfft,1,ngfft,0)
     call sg2002_mpifourdp(1,nfft,ngfft,1,1,& !cplx = 1; ndat = 1;isign = 1, means G -> r
       fftn2_distrib,ffti2_local,fftn3_distrib,ffti3_local,rhog,work,xpaw_mpi_world)
            
     rhor(:,ispden)=work(:)

   end do ! End loop on spins

!  Non-collinear magnetism: avoid zero magnetization, because it produces numerical instabilities
!    Add a small real to the magnetization
   if (all(abs(spinat(:,:))<tol10)) rhor(:,4)=rhor(:,4)+tol14

 end if ! nspden==4

 LIBPAW_DEALLOCATE(length)
 LIBPAW_DEALLOCATE(use_gaussian)
 LIBPAW_DEALLOCATE(spinat_indx)
 LIBPAW_DEALLOCATE(work)
 LIBPAW_DEALLOCATE(rhog)

 contains

!Real and imaginary parts of phase.
   function phr_ini(x1,y1,x2,y2,x3,y3)

   real(dp) :: phr_ini
   real(dp),intent(in) :: x1,x2,x3,y1,y2,y3
   phr_ini=(x1*x2-y1*y2)*x3-(y1*x2+x1*y2)*y3
 end function phr_ini

   function phi_ini(x1,y1,x2,y2,x3,y3)

   real(dp) :: phi_ini
   real(dp),intent(in) :: x1,x2,x3,y1,y2,y3
   phi_ini=(x1*x2-y1*y2)*y3+(y1*x2+x1*y2)*x3
 end function phi_ini

   function ph1_ini(nri,ig1,ia)

   real(dp) :: ph1_ini
   integer,intent(in) :: nri,ig1,ia
   ph1_ini=ph1d(nri,ig1+1+n1+(ia-1)*(2*n1+1))
 end function ph1_ini

   function ph2_ini(nri,ig2,ia)

   real(dp) :: ph2_ini
   integer,intent(in) :: nri,ig2,ia
   ph2_ini=ph1d(nri,ig2+1+n2+(ia-1)*(2*n2+1)+natom*(2*n1+1))
 end function ph2_ini

   function ph3_ini(nri,ig3,ia)

   real(dp) :: ph3_ini
   integer,intent(in) :: nri,ig3,ia
   ph3_ini=ph1d(nri,ig3+1+n3+(ia-1)*(2*n3+1)+natom*(2*n1+1+2*n2+1))
 end function ph3_ini

   function phre_ini(ig1,ig2,ig3,ia)

   real(dp) :: phre_ini
   integer,intent(in) :: ig1,ig2,ig3,ia
   phre_ini=phr_ini(ph1_ini(re,ig1,ia),ph1_ini(im,ig1,ia),&
&   ph2_ini(re,ig2,ia),ph2_ini(im,ig2,ia),ph3_ini(re,ig3,ia),ph3_ini(im,ig3,ia))
 end function phre_ini

   function phimag_ini(ig1,ig2,ig3,ia)

   real(dp) :: phimag_ini
   integer,intent(in) :: ig1,ig2,ig3,ia
   phimag_ini=phi_ini(ph1_ini(re,ig1,ia),ph1_ini(im,ig1,ia),&
&   ph2_ini(re,ig2,ia),ph2_ini(im,ig2,ia),ph3_ini(re,ig3,ia),ph3_ini(im,ig3,ia))
 end function phimag_ini

end subroutine initro
!!***

end module m_mkrho
!!***
