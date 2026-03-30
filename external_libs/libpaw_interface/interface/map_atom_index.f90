! This subroutine processes typat, which is an array that stores the type index of each atom, 
! and generates several other arrays:
! 1. nattyp : this is an array of size ntype, which stores how many atoms there are of each type
! 2. atindx : 'index table' of atoms, which creates an alternative order of atoms based on order of types
! 3. atindx1 : the inverse of atindx

! For example, a CH4 molecule where type 1 is H, type 2 is C, and typat is given by 2 1 1 1 1
! then nattyp is (4 1)
! atindx is (5 1 2 3 4), where the H atoms are ordered before the C atom
! and atindx1 is (2 3 4 5 1), because, for example, '1' is the 2nd element in array atindx

subroutine map_atom_index(ntypat,natom,typat)
    use libpaw_mod
    implicit none

    integer :: ntypat,natom,typat(natom)
    integer :: indx, itypat, iatom

    indx=1
    do itypat = 1, ntypat
        nattyp(itypat)=0
        do iatom = 1, natom
            if(typat(iatom) == itypat)then
                atindx(iatom)=indx
                atindx1(indx)=iatom
                indx=indx+1
                nattyp(itypat)=nattyp(itypat)+1
            end if
        end do
    end do

    !write(*,*) 'nattyp',nattyp
    !write(*,*) 'atindx1',atindx1
end subroutine