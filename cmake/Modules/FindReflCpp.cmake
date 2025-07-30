# FindReflCpp.cmake
# Find the refl-cpp header-only library
#
# This module creates the following imported target:
#  refl-cpp::refl-cpp   - The refl-cpp header-only library target
#
# Usage:
#  find_package(ReflCpp REQUIRED)
#  target_link_libraries(your_target PRIVATE refl-cpp::refl-cpp)
#
# You can specify the refl-cpp installation directory by setting:
#  REFL_CPP_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_refl_cpp_search_paths)

# Priority 1: CMake variable REFL_CPP_ROOT
if(REFL_CPP_ROOT)
    list(APPEND _refl_cpp_search_paths ${REFL_CPP_ROOT})
endif()

# Priority 2: Environment variable REFL_CPP_ROOT
if(DEFINED ENV{REFL_CPP_ROOT})
    list(APPEND _refl_cpp_search_paths $ENV{REFL_CPP_ROOT})
endif()

# Priority 3: Project's dependency location
list(APPEND _refl_cpp_search_paths
    ${CMAKE_SOURCE_DIR}/_build/target-deps/refl-cpp
)

# Priority 4: Standard system locations
list(APPEND _refl_cpp_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(REFL_CPP_INCLUDE_DIR
    NAMES refl.hpp
    PATHS ${_refl_cpp_search_paths}
    PATH_SUFFIXES include
    DOC "refl-cpp include directory"
)

# Extract version information from the header file (if available)
if(REFL_CPP_INCLUDE_DIR AND EXISTS "${REFL_CPP_INCLUDE_DIR}/refl.hpp")
    file(READ "${REFL_CPP_INCLUDE_DIR}/refl.hpp" _refl_cpp_header_contents)

    string(REGEX MATCH "#define REFL_VERSION_MAJOR ([0-9]+)" _refl_cpp_major_match "${_refl_cpp_header_contents}")
    string(REGEX MATCH "#define REFL_VERSION_MINOR ([0-9]+)" _refl_cpp_minor_match "${_refl_cpp_header_contents}")
    string(REGEX MATCH "#define REFL_VERSION_PATCH ([0-9]+)" _refl_cpp_patch_match "${_refl_cpp_header_contents}")

    if(_refl_cpp_major_match)
        set(REFL_CPP_VERSION_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_refl_cpp_minor_match)
        set(REFL_CPP_VERSION_MINOR "${CMAKE_MATCH_1}")
    endif()
    if(_refl_cpp_patch_match)
        set(REFL_CPP_VERSION_PATCH "${CMAKE_MATCH_1}")
    endif()

    if(REFL_CPP_VERSION_MAJOR AND REFL_CPP_VERSION_MINOR AND REFL_CPP_VERSION_PATCH)
        set(REFL_CPP_VERSION "${REFL_CPP_VERSION_MAJOR}.${REFL_CPP_VERSION_MINOR}.${REFL_CPP_VERSION_PATCH}")
    endif()

    unset(_refl_cpp_header_contents)
    unset(_refl_cpp_major_match)
    unset(_refl_cpp_minor_match)
    unset(_refl_cpp_patch_match)
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ReflCpp
    FOUND_VAR REFLCPP_FOUND
    REQUIRED_VARS REFL_CPP_INCLUDE_DIR
    VERSION_VAR REFL_CPP_VERSION
)

# Create imported target for modern CMake
if(REFLCPP_FOUND)
    if(NOT TARGET refl-cpp::refl-cpp)
        add_library(refl-cpp::refl-cpp INTERFACE IMPORTED)
        set_target_properties(refl-cpp::refl-cpp PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${REFL_CPP_INCLUDE_DIR}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_refl_cpp_search_paths)
mark_as_advanced(REFL_CPP_INCLUDE_DIR)