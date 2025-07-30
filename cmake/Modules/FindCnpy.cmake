# FindCnpy.cmake
# Find the Cnpy library
#
# This module creates the following imported target:
#  Cnpy::Cnpy   - The Cnpy library target
#
# Usage:
#  find_package(Cnpy REQUIRED)
#  target_link_libraries(your_target PRIVATE Cnpy::Cnpy)
#
# You can specify the Cnpy installation directory by setting:
#  CNPY_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_cnpy_search_paths)

# Priority 1: CMake variable CNPY_ROOT
if(CNPY_ROOT)
    list(APPEND _cnpy_search_paths ${CNPY_ROOT})
endif()

# Priority 2: Environment variable CNPY_ROOT
if(DEFINED ENV{CNPY_ROOT})
    list(APPEND _cnpy_search_paths $ENV{CNPY_ROOT})
endif()

# Priority 3: Standard system locations
list(APPEND _cnpy_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# For multi-config generators, try to find both debug and release versions
set(_found_debug FALSE)
set(_found_release FALSE)

# Find debug version
find_path(CNPY_INCLUDE_DIR_DEBUG
    NAMES cnpy.h
    PATHS ${_cnpy_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/cnpy/debug
    PATH_SUFFIXES include include/cnpy
    DOC "Cnpy debug include directory"
)

find_library(CNPY_LIBRARY_DEBUG
    NAMES cnpy
    PATHS ${_cnpy_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/cnpy/debug
    PATH_SUFFIXES lib lib64 lib/cnpy
    DOC "Cnpy debug library"
)

if(CNPY_INCLUDE_DIR_DEBUG AND CNPY_LIBRARY_DEBUG)
    set(_found_debug TRUE)
endif()

# Find release version
find_path(CNPY_INCLUDE_DIR_RELEASE
    NAMES cnpy.h
    PATHS ${_cnpy_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/cnpy/release
    PATH_SUFFIXES include include/cnpy
    DOC "Cnpy release include directory"
)

find_library(CNPY_LIBRARY_RELEASE
    NAMES cnpy
    PATHS ${_cnpy_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/cnpy/release
    PATH_SUFFIXES lib lib64 lib/cnpy
    DOC "Cnpy release library"
)

if(CNPY_INCLUDE_DIR_RELEASE AND CNPY_LIBRARY_RELEASE)
    set(_found_release TRUE)
endif()

# Fallback to old behavior if multi-config versions not found
if(NOT _found_debug AND NOT _found_release)
    find_path(CNPY_INCLUDE_DIR
        NAMES cnpy.h
        PATHS ${_cnpy_search_paths}
        PATH_SUFFIXES include include/cnpy
        DOC "Cnpy include directory"
    )

    find_library(CNPY_LIBRARY
        NAMES cnpy
        PATHS ${_cnpy_search_paths}
        PATH_SUFFIXES lib lib64 lib/cnpy
        DOC "Cnpy library"
    )
endif()

# Determine what we found for the standard args check
if(_found_debug OR _found_release)
    set(CNPY_FOUND TRUE)
    if(_found_debug)
        set(CNPY_INCLUDE_DIR ${CNPY_INCLUDE_DIR_DEBUG})
        set(CNPY_LIBRARY ${CNPY_LIBRARY_DEBUG})
    else()
        set(CNPY_INCLUDE_DIR ${CNPY_INCLUDE_DIR_RELEASE})
        set(CNPY_LIBRARY ${CNPY_LIBRARY_RELEASE})
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Cnpy
    FOUND_VAR CNPY_FOUND
    REQUIRED_VARS CNPY_INCLUDE_DIR CNPY_LIBRARY
)

# Create imported target for modern CMake
if(CNPY_FOUND)
    if(NOT TARGET Cnpy::Cnpy)
        add_library(Cnpy::Cnpy UNKNOWN IMPORTED)

        # Use multi-config approach if we found both debug and release
        if(_found_debug AND _found_release)
            # Use generator expressions to select the right version based on config
            set_target_properties(Cnpy::Cnpy PROPERTIES
                IMPORTED_LOCATION_DEBUG "${CNPY_LIBRARY_DEBUG}"
                IMPORTED_LOCATION_RELEASE "${CNPY_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_RELWITHDEBINFO "${CNPY_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_MINSIZEREL "${CNPY_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "$<IF:$<CONFIG:Debug>,${CNPY_INCLUDE_DIR_DEBUG},${CNPY_INCLUDE_DIR_RELEASE}>"
                INTERFACE_LINK_LIBRARIES "ZLIB::ZLIB"
            )
        elseif(_found_debug)
            # Only debug version found
            set_target_properties(Cnpy::Cnpy PROPERTIES
                IMPORTED_LOCATION "${CNPY_LIBRARY_DEBUG}"
                INTERFACE_INCLUDE_DIRECTORIES "${CNPY_INCLUDE_DIR_DEBUG}"
                INTERFACE_LINK_LIBRARIES "ZLIB::ZLIB"
            )
        elseif(_found_release)
            # Only release version found
            set_target_properties(Cnpy::Cnpy PROPERTIES
                IMPORTED_LOCATION "${CNPY_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "${CNPY_INCLUDE_DIR_RELEASE}"
                INTERFACE_LINK_LIBRARIES "ZLIB::ZLIB"
            )
        else()
            # Fallback to old behavior
            set_target_properties(Cnpy::Cnpy PROPERTIES
                IMPORTED_LOCATION "${CNPY_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${CNPY_INCLUDE_DIR}"
                INTERFACE_LINK_LIBRARIES "ZLIB::ZLIB"
            )
        endif()
    endif()
endif()

# Clean up temporary variables
unset(_cnpy_search_paths)
unset(_found_debug)
unset(_found_release)
unset(CNPY_INCLUDE_DIR)
unset(CNPY_LIBRARY)
unset(CNPY_INCLUDE_DIR_DEBUG)
unset(CNPY_LIBRARY_DEBUG)
unset(CNPY_INCLUDE_DIR_RELEASE)
unset(CNPY_LIBRARY_RELEASE)