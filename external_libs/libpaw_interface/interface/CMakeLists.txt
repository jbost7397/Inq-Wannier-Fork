set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL "")
set(FETCHCONTENT_QUIET FALSE)
Include(FetchContent)

message(STATUS "Downloading Catch2 v3p3p2 for gpurun")
FetchContent_Declare(
	Catch2
	GIT_REPOSITORY https://github.com/catchorg/Catch2.git
	GIT_TAG        v3.3.2
  GIT_PROGRESS   TRUE
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)
link_libraries(Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Use PkgConfig to find FFTW
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW fftw3 IMPORTED_TARGET)

if(FFTW_FOUND)
  list(APPEND math_libs PkgConfig::FFTW)
else()  # Fallback if pkg-config does not find FFTW
  if(NOT DEFINED FFTW_LIBRARIES)
    message(WARNING "FFTW not detected by pkg-config, using fftw3 or define -DFFTW_LIBRARIES (e.g. =-lfftw3) and -DFFTW_INCLUDE_DIRS (e.g. =/usr/include)")
    list(APPEND math_libs fftw3)
    include_directories(${FFTW_INCLUDE_DIRS})
  else()
    list(APPEND math_libs ${FFTW_LIBRARIES})
    include_directories(${FFTW_INCLUDE_DIRS})
  endif()
  check_cxx_symbol_exists(fftw_plan_dft_1d fftw3.h FFTW3_CHECKED)
  if(NOT FFTW_FOUND)
    message(FATAL_ERROR "FFTW not found")
  endif()
endif()

# Find LAPACK
find_package(LAPACK REQUIRED)
list(APPEND math_libs LAPACK::LAPACK)

# Handle Fortran libraries based on the compiler
if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  list(APPEND math_libs -lgfortran)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  list(APPEND math_libs -lifcore)
elseif(CMAKE_CXX_COMPILER_ID MATCHES Clang)
  list(APPEND math_libs -lgfortran)
else()
  message(WARNING "Cannot find the correct library for Fortran.")
endif()

#OpenMP causing issues when compiling with Clang and other builds in INQ pipeline
#find_package(OpenMP REQUIRED)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(Threads) 

add_library(libpaw_mod OBJECT
    0_libpaw_mod.f90
    )
target_include_directories(libpaw_mod PUBLIC ${PROJECT_BINARY_DIR})
target_link_libraries(libpaw_mod paw_level3 paw_level4)
add_dependencies(libpaw_mod paw_level3 paw_level4)

add_library(libpaw_interface_files OBJECT
    fortran_main.f90
    get_dij.f90
    get_nhat.f90
    paw_force.f90
    get_vloc_ncoret.f90
    init_rho.f90
    map_atom_index.f90
    prepare_libpaw.f90
    set_rhoij.f90
    read_pawxml_all.f90
    prep_read_pawxml.f90
    extract_rad_info.f90
    pawxml_zval.f90
    pawxml_lmax.f90
    pawxml_local_pp.f90
    pawxml_nchan_max.f90
    pawproj_exist.f90
    pawproj.f90
)
target_include_directories(libpaw_interface_files PUBLIC
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/interface
    ${PROJECT_BINARY_DIR}/abi_add
    ${PROJECT_BINARY_DIR}/abi_xc_lowlevel
    ${PROJECT_BINARY_DIR}/abi_add_paw)
target_link_libraries(libpaw_interface_files libpaw_mod paw_level4 fft_level3 paw_add)
add_dependencies(libpaw_interface_files libpaw_mod paw_level4 fft_level3 paw_add)

# Create INTERFACE library..
add_library(libpaw_interface INTERFACE)
# .. which combines OBJECT libraries
target_sources(libpaw_interface INTERFACE 
    $<TARGET_OBJECTS:paw_defs>
    $<TARGET_OBJECTS:paw_mpi>
    $<TARGET_OBJECTS:paw_tools>
    $<TARGET_OBJECTS:fft_level1>
    $<TARGET_OBJECTS:fft_level2>
    $<TARGET_OBJECTS:fft_level3>
    $<TARGET_OBJECTS:paw_libxc_c>
    $<TARGET_OBJECTS:paw_libxc>
    $<TARGET_OBJECTS:paw_level1>
    $<TARGET_OBJECTS:paw_level2>
    $<TARGET_OBJECTS:paw_level3>
    $<TARGET_OBJECTS:xc_basic>
    $<TARGET_OBJECTS:xc_driver>
    $<TARGET_OBJECTS:paw_xc>
    $<TARGET_OBJECTS:paw_level4>
    $<TARGET_OBJECTS:paw_add>
    $<TARGET_OBJECTS:libpaw_mod>
    $<TARGET_OBJECTS:libpaw_interface_files>)

add_executable(paw_lib main.cpp)
add_dependencies(paw_lib libpaw_interface_files)
target_include_directories(paw_lib PUBLIC ${PROJECT_BINARY_DIR}/interface)
target_link_directories(paw_lib PUBLIC
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/interface
    ${PROJECT_BINARY_DIR}/abi_add
    ${PROJECT_BINARY_DIR}/abi_xc_lowlevel
    ${PROJECT_BINARY_DIR}/abi_add_paw)
target_link_libraries(paw_lib PUBLIC
    libpaw_interface
    ${math_libs}
    Threads::Threads
    #OpenMP::OpenMP_CXX
    MPI::MPI_CXX
    m)

set(TEST_REFERENCE_FILES
  C.xml
  H.xml 
  input  
  pawfiles 
  veff
  rhoij
  reference_data
)

foreach(file ${TEST_REFERENCE_FILES})
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_CURRENT_SOURCE_DIR}/${file}
        ${CMAKE_CURRENT_BINARY_DIR}/${file})
endforeach()

add_executable(libpaw_test test_libpaw.cpp)
target_link_libraries(libpaw_test PRIVATE Catch2::Catch2WithMain)

add_test(NAME run_paw_lib COMMAND ${CMAKE_CURRENT_BINARY_DIR}/paw_lib
        COMMENT "Running paw_lib to generate necessary files for libpaw_test")
add_test(NAME libpaw::unit_test COMMAND ${CMAKE_CURRENT_BINARY_DIR}/libpaw_test)
set_tests_properties(libpaw::unit_test PROPERTIES 
        DEPENDS run_paw_lib
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

