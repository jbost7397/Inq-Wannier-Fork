# -*- indent-tabs-mode: t -*-

add_subdirectory(pseudopod)

message("\nCONFIGURING LIBXC\n")
set(DISABLE_FXC ON)
set(BUILD_TESTING OFF)
add_subdirectory(libxc)
message("\nDONE CONFIGURING LIBXC\n")

message("\nCONFIGURING HEFFTE\n")
string(REPLACE -Werror -Wno-error CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE -pedantic-errors "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(BUILD_GPU ${ENABLE_CUDA})
set(GPU_TARGET "Pascal Volta Turing" CACHE STRING "CUDA architectures to compile for; one or more of Fermi, Kepler, Maxwell, Pascal, Volta, or valid sm_[0-9][0-9]")
add_subdirectory(heffte)
message("\nDONE CONFIGURING HEFFTE\n")
