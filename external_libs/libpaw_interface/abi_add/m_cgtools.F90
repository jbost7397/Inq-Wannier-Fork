!!****m* ABINIT/m_cgtools
!! NAME
!!  m_cgtools
!!
!! FUNCTION
!! This module defines wrappers for BLAS routines. The arguments are stored
!! using the "cg" convention, namely real array of shape cg(2,...)
!!
!! COPYRIGHT
!! Copyright (C) 1992-2022 ABINIT group (MG, MT, XG, DCA, GZ, FB, MVer, DCA, GMR, FF)
!! This file is distributed under the terms of the
!! GNU General Public License, see ~abinit/COPYING
!! or http://www.gnu.org/copyleft/gpl.txt .
!! For the initials of contributors, see ~abinit/doc/developers/contributors.txt .
!!
!! NOTES
!! 1) The convention about names of interfaced routine is: cg_<name>,
!!    where <name> is equal to the name of the standard BLAS routine
!!
!! 2) Blas routines are called without an explicit interface on purpose since
!!
!!    a) The compiler should pass the base address of the array to the F77 BLAS
!!
!!    b) Any compiler would complain about type mismatch (REAL,COMPLEX)
!!       if an explicit interface is given.
!!
!! 3) The use of mpi_type is not allowed here. MPI parallelism should be handled in a generic
!!    way by passing the MPI communicator so that the caller can decide how to handle MPI.
!!

#include "libpaw.h"

module m_cgtools

 USE_DEFS
 USE_MSG_HANDLING
 USE_MPI_WRAPPERS
 USE_MEMORY_PROFILING

 implicit none

 ! Helper functions for DFT calculations.
 public :: mean_fftr                ! Compute the mean of an arraysp(nfft,nspden), over the FFT grid.
!***

CONTAINS  !========================================================================================
!!***

!----------------------------------------------------------------------

!!****f* m_cgtools/mean_fftr
!! NAME
!! mean_fftr
!!
!! FUNCTION
!!  Compute the mean of an arraysp(nfft,nspden), over the FFT grid, for each component nspden,
!!  and return it in meansp(nspden).
!!  Take into account the spread of the array due to parallelism: the actual number of fft
!!  points is nfftot, but the number of points on this proc is nfft only.
!!  So : for ispden from 1 to nspden
!!       meansp(ispden) = sum(ifft=1,nfftot) arraysp(ifft,ispden) / nfftot
!!
!! INPUTS
!!  arraysp(nfft,nspden)=the array whose average has to be computed
!!  nfft=number of FFT points stored by one proc
!!  nfftot=total number of FFT points
!!  nspden=number of spin-density components
!!
!! OUTPUT
!!  meansp(nspden)=mean value for each nspden component
!!
!! SOURCE

subroutine mean_fftr(arraysp,meansp,nfft,nfftot,nspden,mpi_comm_sphgrid)

!Arguments ------------------------------------
!scalars
 integer,intent(in) :: nfft,nfftot,nspden
 integer,intent(in),optional:: mpi_comm_sphgrid
!arrays
 real(dp),intent(in) :: arraysp(nfft,nspden)
 real(dp),intent(out) :: meansp(nspden)

!Local variables-------------------------------
!scalars
 integer :: ierr,ifft,ispden,nproc_sphgrid
 real(dp) :: invnfftot,tmean

! *************************************************************************

 invnfftot=one/(dble(nfftot))

 do ispden=1,nspden
   tmean=zero
!$OMP PARALLEL DO REDUCTION(+:tmean)
   do ifft=1,nfft
     tmean=tmean+arraysp(ifft,ispden)
   end do
   meansp(ispden)=tmean*invnfftot
 end do

!XG030514 : MPIWF The values of meansp(ispden) should
!now be summed accross processors in the same WF group, and spread on all procs.
 if(present(mpi_comm_sphgrid)) then
   nproc_sphgrid=xpaw_mpi_comm_size(mpi_comm_sphgrid)
   if(nproc_sphgrid>1) then
     ! This case is not called in ABINIT
     !call xpaw_mpi_sum(meansp,nspden,mpi_comm_sphgrid,ierr)
   end if
 end if

end subroutine mean_fftr
!!***

end module m_cgtools
!!***
