project(libpaw_interface)
enable_language(Fortran)
cmake_minimum_required(VERSION 3.16)

enable_testing()
include(CTest)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(paw_defs OBJECT m_libpaw_defs.F90)

add_library(paw_mpi OBJECT m_libpaw_mpi.F90)
add_dependencies(paw_mpi paw_defs)

add_library(paw_tools OBJECT m_libpaw_tools.F90)
target_link_libraries(paw_tools paw_defs paw_mpi)
add_dependencies(paw_tools paw_defs paw_mpi)

add_library(paw_libxc_c OBJECT libpaw_libxc.c)
add_library(paw_libxc OBJECT m_libpaw_libxc.F90)
target_link_libraries(paw_libxc paw_defs paw_mpi paw_tools)
add_dependencies(paw_libxc paw_defs paw_mpi paw_tools)

add_library(paw_level1
    OBJECT
    m_paw_numeric.F90
    m_paral_atom.F90
    m_paw_io.F90
    m_paw_sphharm.F90
    m_pawtab.F90)
target_link_libraries(paw_level1 paw_defs paw_mpi paw_tools)
add_dependencies(paw_level1 paw_defs paw_mpi paw_tools)

add_library(paw_level2
    OBJECT
    m_pawang.F90
    m_pawrad.F90
    m_paw_finegrid.F90
    m_paw_ij.F90
    m_pawcprj.F90
    m_pawfgrtab.F90)
target_link_libraries(paw_level2 paw_level1 paw_defs paw_mpi paw_tools)
add_dependencies(paw_level2 paw_level1 paw_defs paw_mpi paw_tools)

add_library(paw_level3
    OBJECT
    m_pawxmlps.F90
    m_paw_an.F90
    m_paw_gaussfit.F90
    m_pawrhoij.F90
    m_paw_atom.F90)
target_link_libraries(paw_level3 paw_level2 paw_level1 paw_defs paw_mpi paw_tools)
add_dependencies(paw_level3 paw_level2 paw_level1 paw_defs paw_mpi paw_tools)

add_subdirectory(abi_xc_lowlevel)
add_subdirectory(abi_add)

add_compile_definitions(HAVE_XC_ABINIT=1)
add_library(paw_xc OBJECT
    m_pawxc.F90)
target_include_directories(paw_xc PUBLIC ${PROJECT_BINARY_DIR}/abi_xc_lowlevel)
target_link_libraries(paw_xc paw_level2 paw_libxc xc_driver)
add_dependencies(paw_xc paw_level2 paw_libxc xc_driver)

add_library(paw_level4 OBJECT
    m_pawpsp.F90
    m_pawdij.F90)
target_link_libraries(paw_level4 paw_xc paw_level3)
add_dependencies(paw_level4 paw_xc paw_level3)

add_subdirectory(abi_add_paw)
add_subdirectory(interface)
