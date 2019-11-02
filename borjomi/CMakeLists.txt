###############################################################
###### Set minimum version of CMake. Since command ############
###### 'project' use VERSION sub-option we need at least 3.0. #
###############################################################

cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

###############################################################
###### Set variables: #########################################
######   * PROJECT_NAME #######################################
######   * PROJECT_VERSION ####################################
###############################################################

project(borjomi VERSION 1.0.0 LANGUAGES C CXX)

###############################################################
###### Enables link_directories() treat paths relative ########
###### to the source dir. #####################################
###############################################################

if(POLICY CMP0015)
    cmake_policy(SET CMP0015 NEW)
endif(POLICY CMP0015)

###############################################################
### Enables project() command manages VERSION variables. ######
###############################################################

if(POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif(POLICY CMP0048)

###############################################################
###### Change the default build type from Debug to Release ####
###############################################################

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type specified; defaulting to CMAKE_BUILD_TYPE=Release.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
else(NOT CMAKE_BUILD_TYPE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("==========================================================================================")
        message(STATUS "Build type: Debug. Performance will be terrible!")
        message(STATUS "Add -DCMAKE_BUILD_TYPE=Release to the CMake command line to get an optimized build.")
        message("==========================================================================================")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

###############################################################
###### Define user options ####################################
###############################################################

option(USE_AVX        "Build borjomi with AVX library support"            OFF)
option(USE_AVX2       "Build borjomi with AVX2 library support"           OFF)
option(USE_INTEL_MKL  "Build borjomi with Intel MKL library support"      OFF)
option(USE_THREADS    "Build borjomi with pthread support"                OFF)
option(USE_CUDA       "Build borjomi with OpenCL library support"         OFF)

option(BUILD_EXAMPLES "Set to ON to build examples"                         ON)

###############################################################
###### Create the library target ##############################
###############################################################

set(project_library_target_name ${PROJECT_NAME})
set(PACKAGE_NAME Borjomi)
add_library(${project_library_target_name} INTERFACE)

###############################################################
###### Set C++ version ########################################
###############################################################

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++14 support has been enabled by default.")

###############################################################
### Setup the optional dependencies ###########################
###############################################################

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

###############################################################
###### Find Intel MKL: Intel's CBLAS implementation ###########
###############################################################

if(USE_INTEL_MKL)
    find_package(INTELMKL REQUIRED)
    if(INTELMKL_FOUND)
        message(STATUS "Found Intel MKL: ${INTEL_MKL_INCLUDE_DIR}")
        add_definitions(-DCNN_USE_INTEL_MKL)
        include_directories(SYSTEM ${INTEL_MKL_INCLUDE_DIR})
        list(APPEND REQUIRED_LIBRARIES "-Wl,--start-group ${INTEL_MKL_LIB_1} ${INTEL_MKL_LIB_2} ${INTEL_MKL_LIB_3} -Wl,--end-group -ldl")
    else(INTELMKL_FOUND)
        message(FATAL_ERROR "Can't find Intel MKL. Please set MKLROOT")
    endif()
endif(USE_INTEL_MKL)

###############################################################
###### Standard C++11 multithread support #####################
###############################################################

if(USE_THREADS)
  if(NOT WIN32)
    set(USE_PTHREAD ON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
    message(STATUS "Using Pthread")
  else(NOT WIN32)
    set(USE_PTHREAD OFF)
  endif(NOT WIN32)
endif(USE_THREADS)

###############################################################
###### Speed up build with cotire #############################
###############################################################

include(cotire)

###############################################################
###### Check and set compiler flags for Unix based systems ####
###############################################################

if(CMAKE_COMPILER_IS_GNUCXX OR MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx"  COMPILER_HAS_AVX_FLAG)
    check_cxx_compiler_flag("-mavx2" COMPILER_HAS_AVX2_FLAG)
    check_cxx_compiler_flag("-mfma" COMPILER_HAS_AVX2_FLAG)

    #############################################
    ### Advanced Vector Extensions (AVX) ########
    #############################################

    if(USE_AVX AND COMPILER_HAS_AVX_FLAG)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx")
    endif(USE_AVX AND COMPILER_HAS_AVX_FLAG)

    #############################################
    ### Advanced Vector Extensions 2 (AVX2) #####
    #############################################

    if(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -mavx2 -mfma -march=core-avx2")
    endif(USE_AVX2 AND COMPILER_HAS_AVX2_FLAG)

    #############################################
    ### Other flags #############################
    #############################################

    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing -Wno-deprecated")
    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} -O3")
    set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pthread")

endif()

###############################################################
###### Check compiler flags for Windows systems ###############
###############################################################

if(MSVC)

    #############################################
    ### Advanced Vector Extensions (AVX) ########
    #############################################

    if(USE_AVX)
        add_definitions(-DCNN_USE_AVX)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX")
    endif(USE_AVX)

    #############################################
    ### Advanced Vector Extensions 2 (AVX2) #####
    #############################################

    if(USE_AVX2)
        add_definitions(-DCNN_USE_AVX2)
        set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /arch:AVX2")
    endif(USE_AVX2)

    #############################################
    ### Other flags #############################
    #############################################

    set(EXTRA_C_FLAGS_RELEASE "${EXTRA_C_FLAGS_RELEASE} /Ox /Oi /Ot /Oy /GL /fp:fast /GS-")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(EXTRA_C_FLAGS_DEBUG "${EXTRA_C_FLAGS_DEBUG}")
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /W4 /bigobj")

    # this is fine
    add_definitions(-D _CRT_SECURE_NO_WARNINGS)
    add_definitions(-D _SCL_SECURE_NO_WARNINGS)
    # prolly powerless with header-only project
    set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} /MP")

endif()

###############################################################
###### Compiler options #######################################
###############################################################

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${EXTRA_C_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG}")

if(PROFILE AND (CMAKE_COMPILER_IS_GNUCXX OR MINGW OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
endif()

###############################################################
###### CMake config files #####################################
###############################################################

string(REGEX REPLACE "_" "" PROJECT_NAME_JOINED ${PROJECT_NAME})

set(cmake_conf_version_file "${PROJECT_NAME_JOINED}-config-version.cmake")
set(cmake_targets_file      "${PROJECT_NAME_JOINED}-targets.cmake")

set(targets_export_name "${PROJECT_NAME_JOINED}-targets")
set(namespace "${PACKAGE_NAME}::")
    
##############################################################
###### Set up install directories. INCLUDE_INSTALL_DIR and ###
###### CMAKECONFIG_INSTALL_DIR must not be absolute paths. ###
##############################################################

if(WIN32)
    set(include_install_dir Include)
    set(include_install_dir_full Include)
    set(config_install_dir CMake)
elseif(UNIX)
    set(include_install_dir include)
    set(include_install_dir_postfix "${project_library_target_name}")
    set(include_install_dir_full    "${include_install_dir}/${include_install_dir_postfix}")
    set(config_install_dir share/${PACKAGE_NAME})
else()
    message(FATAL_ERROR "Not supported system type. Options: UNIX or WIN32.")
endif()

##############################################################
###### Configure library target ##############################
##############################################################

target_include_directories(
    ${project_library_target_name} INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${include_install_dir_full}>)

## include(CMakePackageConfigHelpers)
## configure_package_config_file(
##     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Templates/${cmake_conf_file}.in"
##     "${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_file}"
##     PATH_VARS include_install_dir_full
##     INSTALL_DESTINATION ${config_install_dir})
## 
## write_basic_package_version_file(
##     ${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_version_file}
##     VERSION ${PROJECT_VERSION}
##     COMPATIBILITY SameMajorVersion)

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
# Create *-targets.cmake file for build directory
install(TARGETS ${project_library_target_name}
        EXPORT  ${targets_export_name}
        INCLUDES DESTINATION ${include_install_dir})

export(EXPORT ${targets_export_name}
       FILE   ${CMAKE_CURRENT_BINARY_DIR}/${cmake_targets_file})

# Install *-targets.cmake file
install(EXPORT      ${targets_export_name}
        NAMESPACE   ${namespace}
        DESTINATION ${config_install_dir})

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${cmake_conf_version_file}"
    "${PROJECT_SOURCE_DIR}/cmake/Modules/find_intel_mkl.cmake"
    DESTINATION ${config_install_dir} COMPONENT cmake)

# Install headers
install(DIRECTORY   ${PROJECT_SOURCE_DIR}/${project_library_target_name}
        DESTINATION ${include_install_dir})

### Check if protobuf available
## include(cmake/protoc.cmake)

# Subdirectories for examples, testing and documentation
# TODO: explain in brief about different examples, test and docs.
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif(BUILD_EXAMPLES)

####
# Configuration summary
include(cmake/summary.cmake)
printConfigurationSummary()


