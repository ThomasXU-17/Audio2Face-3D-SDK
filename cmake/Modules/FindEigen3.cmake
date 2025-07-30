# FindEigen.cmake
# Find the Eigen header-only library
#
# This module creates the following imported target:
#  Eigen3::Eigen   - The Eigen header-only library target
#
# Usage:
#  find_package(Eigen3 REQUIRED)
#  target_link_libraries(your_target PRIVATE Eigen3::Eigen)
#
# You can specify the Eigen3 installation directory by setting:
#  EIGEN3_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_eigen3_search_paths)

# Priority 1: CMake variable EIGEN_ROOT
if(EIGEN3_ROOT)
    list(APPEND _eigen3_search_paths ${EIGEN3_ROOT})
endif()

# Priority 2: Environment variable EIGEN_ROOT
if(DEFINED ENV{EIGEN3_ROOT})
    list(APPEND _eigen3_search_paths $ENV{EIGEN3_ROOT})
endif()

# Priority 3: Project's dependency location
list(APPEND _eigen3_search_paths
    ${CMAKE_SOURCE_DIR}/_build/target-deps/eigen
)

# Priority 4: Standard system locations
list(APPEND _eigen3_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    PATHS ${_eigen3_search_paths}
    PATH_SUFFIXES include include/eigen3
    DOC "Eigen include directory"
)

# Extract version information from the header file
if(EIGEN3_INCLUDE_DIR AND EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h")
    file(READ "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" _eigen_header_contents)

    string(REGEX MATCH "#define EIGEN_WORLD_VERSION ([0-9]+)" _eigen_world_match "${_eigen_header_contents}")
    string(REGEX MATCH "#define EIGEN_MAJOR_VERSION ([0-9]+)" _eigen_major_match "${_eigen_header_contents}")
    string(REGEX MATCH "#define EIGEN_MINOR_VERSION ([0-9]+)" _eigen_minor_match "${_eigen_header_contents}")

    if(_eigen_world_match)
        set(EIGEN3_VERSION_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_eigen_major_match)
        set(EIGEN3_VERSION_MINOR "${CMAKE_MATCH_1}")
    endif()
    if(_eigen_minor_match)
        set(EIGEN3_VERSION_PATCH "${CMAKE_MATCH_1}")
    endif()

    if(EIGEN3_VERSION_MAJOR AND EIGEN3_VERSION_MINOR AND EIGEN3_VERSION_PATCH)
        set(EIGEN3_VERSION "${EIGEN3_VERSION_MAJOR}.${EIGEN3_VERSION_MINOR}.${EIGEN3_VERSION_PATCH}")
    endif()

    unset(_eigen_header_contents)
    unset(_eigen_world_match)
    unset(_eigen_major_match)
    unset(_eigen_minor_match)
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3
    FOUND_VAR EIGEN3_FOUND
    REQUIRED_VARS EIGEN3_INCLUDE_DIR
    VERSION_VAR EIGEN3_VERSION
)

# Create imported target for modern CMake
if(EIGEN3_FOUND)
    if(NOT TARGET Eigen3::Eigen)
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_eigen3_search_paths)
unset(EIGEN3_INCLUDE_DIR)
unset(EIGEN3_VERSION_MAJOR)
unset(EIGEN3_VERSION_MINOR)
unset(EIGEN3_VERSION_PATCH)