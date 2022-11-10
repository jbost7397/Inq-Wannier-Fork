file(GLOB_RECURSE PRJ_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

if(ENABLE_CUDA)
  set_source_files_properties(${PRJ_SRCS} PROPERTIES LANGUAGE CUDA)
endif()

foreach(PRJ_SOURCE ${PRJ_SRCS})
  get_filename_component(PRJ_EXE ${PRJ_SOURCE} NAME_WE  )
  get_filename_component(PRJ_DIR ${PRJ_SOURCE} DIRECTORY)

  add_executable("${PRJ_DIR}_${PRJ_EXE}" ${PRJ_SOURCE})

  set_target_properties("${PRJ_DIR}_${PRJ_EXE}" PROPERTIES OUTPUT_NAME              ${PRJ_EXE})
  set_target_properties("${PRJ_DIR}_${PRJ_EXE}" PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PRJ_DIR})

  target_include_directories("${PRJ_DIR}_${PRJ_EXE}" PRIVATE ../external_libs)
  target_include_directories("${PRJ_DIR}_${PRJ_EXE}" PRIVATE ../src)
endforeach()
