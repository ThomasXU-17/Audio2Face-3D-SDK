# FindTbtSVD.cmake
# Find the tbtSVD header-only library
#
# This module creates the following imported target:
#  tbtSVD::tbtSVD   - The tbtSVD header-only library target
#
# Usage:
#  find_package(TbtSVD REQUIRED)
#  target_link_libraries(your_target PRIVATE tbtSVD::tbtSVD)
#
# You can specify the tbtSVD installation directory by setting:
#  TBT_SVD_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_tbt_svd_search_paths)

# Priority 1: CMake variable TBT_SVD_ROOT
if(TBT_SVD_ROOT)
    list(APPEND _tbt_svd_search_paths ${TBT_SVD_ROOT})
endif()

# Priority 2: Environment variable TBT_SVD_ROOT
if(DEFINED ENV{TBT_SVD_ROOT})
    list(APPEND _tbt_svd_search_paths $ENV{TBT_SVD_ROOT})
endif()

# Priority 3: Project's dependency location
list(APPEND _tbt_svd_search_paths
    ${CMAKE_SOURCE_DIR}/_build/target-deps/tbtsvd
)

# Priority 4: Standard system locations
list(APPEND _tbt_svd_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(TBT_SVD_INCLUDE_DIR
    NAMES tbtSVD/SVD.h
    PATHS ${_tbt_svd_search_paths}
    PATH_SUFFIXES include
    DOC "tbtSVD include directory"
)

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TbtSVD
    FOUND_VAR TBTSVD_FOUND
    REQUIRED_VARS TBT_SVD_INCLUDE_DIR
)

# Create imported target for modern CMake
if(TBTSVD_FOUND)
    if(NOT TARGET tbtSVD::tbtSVD)
        add_library(tbtSVD::tbtSVD INTERFACE IMPORTED)
        set_target_properties(tbtSVD::tbtSVD PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TBT_SVD_INCLUDE_DIR}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_tbt_svd_search_paths)
mark_as_advanced(TBT_SVD_INCLUDE_DIR)