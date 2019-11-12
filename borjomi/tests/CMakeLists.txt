
#######################################
### Changing path format if needed ####
#######################################

if(MSVC)
    file(GLOB_RECURSE borjomi_headers "${CMAKE_SOURCE_DIR}/borjomi/*.h")
    file(RELATIVE_PATH borjomi_relpath ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_SOURCE_DIR}/borjomi")
    string(REPLACE "/" "\\" borjomi_relpath "${borjomi_relpath}")
    foreach(_source IN ITEMS ${borjomi_headers})
        file(RELATIVE_PATH _source_rel ${CMAKE_CURRENT_SOURCE_DIR} "${_source}")
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        string(REPLACE ${borjomi_relpath} "borjomi" _source_path_msvc ${_source_path_msvc})
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endif()

#######################################
### Adding examples ###################
#######################################

add_executable(test_conv_layer convolutional_layer_tests/convolutional_layer_test.cpp ${borjomi_headers})
target_link_libraries(test_conv_layer ${project_library_target_name} ${REQUIRED_LIBRARIES})        