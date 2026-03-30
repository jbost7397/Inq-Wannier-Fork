add_library(fft_level1 OBJECT
    m_cgtools.F90
    m_fft.F90
    m_fftcore.F90
    defs_fftdata.F90)

target_include_directories(fft_level1 PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(fft_level1 paw_tools)
add_dependencies(fft_level1 paw_tools)

add_library(fft_level2 OBJECT
    m_kg.F90
    m_sg2002.F90)

target_include_directories(fft_level2 PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(fft_level2 fft_level1)
add_dependencies(fft_level2 fft_level1)

add_library(fft_level3 OBJECT
    m_atm2fft.F90
    m_mkrho.F90)

target_include_directories(fft_level3 PUBLIC ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(fft_level3 fft_level2 paw_level1)  
add_dependencies(fft_level3 fft_level2 paw_level1)  