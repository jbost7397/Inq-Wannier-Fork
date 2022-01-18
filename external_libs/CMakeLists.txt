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
add_subdirectory(heffte)
message("\nDONE CONFIGURING HEFFTE\n")
