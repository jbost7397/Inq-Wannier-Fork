# libpaw_interface
Trying to write an inteface for the libpaw library from ABINIT, which can be transferred more easily to other SCF softwares

## Acknowledgments
The subroutines are mostly gathered from 42_libpaw and 65_paw from the ABINIT software. I (Wenfei) have made some modifications so that this module can be compiled individually


## Test
To compile this interface do the following

    mkdir build
    cd build
    cmake ..
    cmake --build .

To run the test program in fortran_main.f90, go into build/test and run paw_lib. 
This should output dij_calc, vloc_calc.dat, ncoret_calc.dat, and nhat_calc.dat which one can compare to the corresponding files in the reference_data directory.
