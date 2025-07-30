# FindMagicEnum.cmake
# Find the magic_enum header-only library
#
# This module creates the following imported target:
#  magic_enum::magic_enum   - The magic_enum header-only library target
#
# Usage:
#  find_package(MagicEnum REQUIRED)
#  target_link_libraries(your_target PRIVATE magic_enum::magic_enum)
#
# You can specify the magic_enum installation directory by setting:
#  MAGIC_ENUM_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_magic_enum_search_paths)

# Priority 1: CMake variable MAGIC_ENUM_ROOT
if(MAGIC_ENUM_ROOT)
    list(APPEND _magic_enum_search_paths ${MAGIC_ENUM_ROOT})
endif()

# Priority 2: Environment variable MAGIC_ENUM_ROOT
if(DEFINED ENV{MAGIC_ENUM_ROOT})
    list(APPEND _magic_enum_search_paths $ENV{MAGIC_ENUM_ROOT})
endif()

# Priority 3: Project's dependency location
list(APPEND _magic_enum_search_paths
    ${CMAKE_SOURCE_DIR}/_build/target-deps/magic_enum
)

# Priority 4: Standard system locations
list(APPEND _magic_enum_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(MAGIC_ENUM_INCLUDE_DIR
    NAMES magic_enum/magic_enum.hpp
    PATHS ${_magic_enum_search_paths}
    PATH_SUFFIXES include
    DOC "magic_enum include directory"
)

# Extract version information from the header file (if available)
if(MAGIC_ENUM_INCLUDE_DIR AND EXISTS "${MAGIC_ENUM_INCLUDE_DIR}/magic_enum/magic_enum.hpp")
    file(READ "${MAGIC_ENUM_INCLUDE_DIR}/magic_enum/magic_enum.hpp" _magic_enum_header_contents)

    string(REGEX MATCH "#define MAGIC_ENUM_VERSION_MAJOR ([0-9]+)" _magic_enum_major_match "${_magic_enum_header_contents}")
    string(REGEX MATCH "#define MAGIC_ENUM_VERSION_MINOR ([0-9]+)" _magic_enum_minor_match "${_magic_enum_header_contents}")
    string(REGEX MATCH "#define MAGIC_ENUM_VERSION_PATCH ([0-9]+)" _magic_enum_patch_match "${_magic_enum_header_contents}")

    if(_magic_enum_major_match)
        set(MAGIC_ENUM_VERSION_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_magic_enum_minor_match)
        set(MAGIC_ENUM_VERSION_MINOR "${CMAKE_MATCH_1}")
    endif()
    if(_magic_enum_patch_match)
        set(MAGIC_ENUM_VERSION_PATCH "${CMAKE_MATCH_1}")
    endif()

    if(MAGIC_ENUM_VERSION_MAJOR AND MAGIC_ENUM_VERSION_MINOR AND MAGIC_ENUM_VERSION_PATCH)
        set(MAGIC_ENUM_VERSION "${MAGIC_ENUM_VERSION_MAJOR}.${MAGIC_ENUM_VERSION_MINOR}.${MAGIC_ENUM_VERSION_PATCH}")
    endif()

    unset(_magic_enum_header_contents)
    unset(_magic_enum_major_match)
    unset(_magic_enum_minor_match)
    unset(_magic_enum_patch_match)
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MagicEnum
    FOUND_VAR MAGICENUM_FOUND
    REQUIRED_VARS MAGIC_ENUM_INCLUDE_DIR
    VERSION_VAR MAGIC_ENUM_VERSION
)

# Create imported target for modern CMake
if(MAGICENUM_FOUND)
    if(NOT TARGET magic_enum::magic_enum)
        add_library(magic_enum::magic_enum INTERFACE IMPORTED)
        set_target_properties(magic_enum::magic_enum PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${MAGIC_ENUM_INCLUDE_DIR}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_magic_enum_search_paths)
mark_as_advanced(MAGIC_ENUM_INCLUDE_DIR)