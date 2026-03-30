add_library(xc_basic OBJECT
	xc_m_xcpositron.F90
	xc_m_xcpbe.F90
	xc_m_xclda.F90
	xc_m_xciit.F90
	xc_m_xchcth.F90
	xc_m_libxc_functionals.F90
	xc_libxc_tools.c
	xc_invcb.F90
	xc_m_xc_noncoll.F90)

target_include_directories(xc_basic PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(xc_basic paw_tools)
add_dependencies(xc_basic paw_tools)

add_library(xc_driver OBJECT
	m_drivexc.F90)
target_include_directories(xc_driver PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(xc_driver xc_basic)
add_dependencies(xc_driver xc_basic)
