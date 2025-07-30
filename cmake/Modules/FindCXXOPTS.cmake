# FindCXXOPTS.cmake
# Find the CXXOPTS header-only library
#
# This module creates the following imported target:
#  CXXOPTS::CXXOPTS   - The CXXOPTS header-only library target
#
# Usage:
#  find_package(CXXOPTS REQUIRED)
#  target_link_libraries(your_target PRIVATE CXXOPTS::CXXOPTS)
#
# You can specify the CXXOPTS installation directory by setting:
#  CXXOPTS_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_cxxopts_search_paths)

# Priority 1: CMake variable CXXOPTS_ROOT
if(CXXOPTS_ROOT)
    list(APPEND _cxxopts_search_paths ${CXXOPTS_ROOT})
endif()

# Priority 2: Environment variable CXXOPTS_ROOT
if(DEFINED ENV{CXXOPTS_ROOT})
    list(APPEND _cxxopts_search_paths $ENV{CXXOPTS_ROOT})
endif()

# Priority 3: Project's dependency location
list(APPEND _cxxopts_search_paths
    ${CMAKE_SOURCE_DIR}/_build/target-deps/cxxopts
)

# Priority 4: Standard system locations
list(APPEND _cxxopts_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(CXXOPTS_INCLUDE_DIR
    NAMES cxxopts.hpp
    PATHS ${_cxxopts_search_paths}
    PATH_SUFFIXES include include/cxxopts
    DOC "CXXOPTS include directory"
)

# Extract version information from the header file
if(CXXOPTS_INCLUDE_DIR AND EXISTS "${CXXOPTS_INCLUDE_DIR}/cxxopts.hpp")
    file(READ "${CXXOPTS_INCLUDE_DIR}/cxxopts.hpp" _cxxopts_header_contents)

    string(REGEX MATCH "#define CXXOPTS__VERSION_MAJOR ([0-9]+)" _cxxopts_major_match "${_cxxopts_header_contents}")
    string(REGEX MATCH "#define CXXOPTS__VERSION_MINOR ([0-9]+)" _cxxopts_minor_match "${_cxxopts_header_contents}")
    string(REGEX MATCH "#define CXXOPTS__VERSION_PATCH ([0-9]+)" _cxxopts_patch_match "${_cxxopts_header_contents}")

    if(_cxxopts_major_match)
        set(CXXOPTS_VERSION_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_cxxopts_minor_match)
        set(CXXOPTS_VERSION_MINOR "${CMAKE_MATCH_1}")
    endif()
    if(_cxxopts_patch_match)
        set(CXXOPTS_VERSION_PATCH "${CMAKE_MATCH_1}")
    endif()

    if(CXXOPTS_VERSION_MAJOR AND CXXOPTS_VERSION_MINOR AND CXXOPTS_VERSION_PATCH)
        set(CXXOPTS_VERSION "${CXXOPTS_VERSION_MAJOR}.${CXXOPTS_VERSION_MINOR}.${CXXOPTS_VERSION_PATCH}")
    endif()

    unset(_cxxopts_header_contents)
    unset(_cxxopts_major_match)
    unset(_cxxopts_minor_match)
    unset(_cxxopts_patch_match)
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CXXOPTS
    FOUND_VAR CXXOPTS_FOUND
    REQUIRED_VARS CXXOPTS_INCLUDE_DIR
    VERSION_VAR CXXOPTS_VERSION
)

# Create imported target for modern CMake
if(CXXOPTS_FOUND)
    if(NOT TARGET CXXOPTS::CXXOPTS)
        add_library(CXXOPTS::CXXOPTS INTERFACE IMPORTED)
        set_target_properties(CXXOPTS::CXXOPTS PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CXXOPTS_INCLUDE_DIR}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_cxxopts_search_paths)
unset(CXXOPTS_INCLUDE_DIR)
unset(CXXOPTS_VERSION_MAJOR)
unset(CXXOPTS_VERSION_MINOR)
unset(CXXOPTS_VERSION_PATCH)