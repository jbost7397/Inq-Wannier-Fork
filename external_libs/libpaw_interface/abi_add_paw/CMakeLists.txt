add_library(paw_add OBJECT
    m_paw_denpot.F90
    m_paw_init.F90
    m_paw_nhat.F90
    m_paw_occupancies.F90
    m_paw_dfpt.F90)

target_include_directories(paw_add PUBLIC 
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/abi_add)
target_link_libraries(paw_add fft_level1 paw_level4)
add_dependencies(paw_add fft_level1 paw_level4)