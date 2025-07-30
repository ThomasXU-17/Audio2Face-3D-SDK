# FindNlohmannJson.cmake
# Find the nlohmann_json header-only library
#
# This module creates the following imported target:
#  nlohmann_json::nlohmann_json   - The nlohmann_json header-only library target
#
# Usage:
#  find_package(NlohmannJson REQUIRED)
#  target_link_libraries(your_target PRIVATE nlohmann_json::nlohmann_json)
#
# You can specify the nlohmann_json installation directory by setting:
#  NLOHMANN_JSON_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_nlohmann_json_search_paths)

# Priority 1: CMake variable NLOHMANN_JSON_ROOT
if(NLOHMANN_JSON_ROOT)
    list(APPEND _nlohmann_json_search_paths ${NLOHMANN_JSON_ROOT})
endif()

# Priority 2: Environment variable NLOHMANN_JSON_ROOT
if(DEFINED ENV{NLOHMANN_JSON_ROOT})
    list(APPEND _nlohmann_json_search_paths $ENV{NLOHMANN_JSON_ROOT})
endif()

# Priority 3: Project's dependency location
list(APPEND _nlohmann_json_search_paths
    ${CMAKE_SOURCE_DIR}/_build/target-deps/nlohmann_json
)

# Priority 4: Standard system locations
list(APPEND _nlohmann_json_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(NLOHMANN_JSON_INCLUDE_DIR
    NAMES nlohmann/json.hpp
    PATHS ${_nlohmann_json_search_paths}
    PATH_SUFFIXES include
    DOC "nlohmann_json include directory"
)

# Extract version information from the header file
if(NLOHMANN_JSON_INCLUDE_DIR AND EXISTS "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp")
    file(READ "${NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp" _nlohmann_json_header_contents)

    string(REGEX MATCH "#define NLOHMANN_JSON_VERSION_MAJOR ([0-9]+)" _nlohmann_json_major_match "${_nlohmann_json_header_contents}")
    string(REGEX MATCH "#define NLOHMANN_JSON_VERSION_MINOR ([0-9]+)" _nlohmann_json_minor_match "${_nlohmann_json_header_contents}")
    string(REGEX MATCH "#define NLOHMANN_JSON_VERSION_PATCH ([0-9]+)" _nlohmann_json_patch_match "${_nlohmann_json_header_contents}")

    if(_nlohmann_json_major_match)
        set(NLOHMANN_JSON_VERSION_MAJOR "${CMAKE_MATCH_1}")
    endif()
    if(_nlohmann_json_minor_match)
        set(NLOHMANN_JSON_VERSION_MINOR "${CMAKE_MATCH_1}")
    endif()
    if(_nlohmann_json_patch_match)
        set(NLOHMANN_JSON_VERSION_PATCH "${CMAKE_MATCH_1}")
    endif()

    if(NLOHMANN_JSON_VERSION_MAJOR AND NLOHMANN_JSON_VERSION_MINOR AND NLOHMANN_JSON_VERSION_PATCH)
        set(NLOHMANN_JSON_VERSION "${NLOHMANN_JSON_VERSION_MAJOR}.${NLOHMANN_JSON_VERSION_MINOR}.${NLOHMANN_JSON_VERSION_PATCH}")
    endif()

    unset(_nlohmann_json_header_contents)
    unset(_nlohmann_json_major_match)
    unset(_nlohmann_json_minor_match)
    unset(_nlohmann_json_patch_match)
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NlohmannJson
    FOUND_VAR NLOHMANNJSON_FOUND
    REQUIRED_VARS NLOHMANN_JSON_INCLUDE_DIR
    VERSION_VAR NLOHMANN_JSON_VERSION
)

# Create imported target for modern CMake
if(NLOHMANNJSON_FOUND)
    if(NOT TARGET nlohmann_json::nlohmann_json)
        add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
        set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE_DIR}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_nlohmann_json_search_paths)
mark_as_advanced(NLOHMANN_JSON_INCLUDE_DIR)