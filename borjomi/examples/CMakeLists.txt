
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

add_executable(example_sinus_fit sinus_fit/sinus_fit.cpp ${borjomi_headers})
target_link_libraries(example_sinus_fit
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_cifar_train cifar10/train.cpp ${borjomi_headers})
target_link_libraries(example_cifar_train
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_mnist_train mnist/train.cpp ${borjomi_headers})
target_link_libraries(example_mnist_train
        ${project_library_target_name} ${REQUIRED_LIBRARIES})

add_executable(example_test_multiplication engine_test/testMultiplication.cpp ${borjomi_headers})
target_link_libraries(example_test_multiplication 
        ${project_library_target_name} ${REQUIRED_LIBRARIES})