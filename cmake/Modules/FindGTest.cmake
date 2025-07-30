# FindGTest.cmake
# Find the Google Test library
#
# This module creates the following imported targets:
#  GTest::gtest      - The main GTest library target
#  GTest::gtest_main - The GTest main library target
#  GTest::gmock      - The GMock library target (optional)
#  GTest::gmock_main - The GMock main library target (optional)
#
# Usage:
#  find_package(GTest REQUIRED)
#  target_link_libraries(your_target PRIVATE GTest::gtest GTest::gtest_main)
#
# You can specify the GTest installation directory by setting:
#  GTEST_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_gtest_search_paths)

# Priority 1: CMake variable GTEST_ROOT
if(GTEST_ROOT)
    list(APPEND _gtest_search_paths ${GTEST_ROOT})
endif()

# Priority 2: Environment variable GTEST_ROOT
if(DEFINED ENV{GTEST_ROOT})
    list(APPEND _gtest_search_paths $ENV{GTEST_ROOT})
endif()

# Priority 3: Standard system locations
list(APPEND _gtest_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# For multi-config generators, try to find both debug and release versions
set(_found_debug FALSE)
set(_found_release FALSE)

# Find debug version
find_path(GTEST_INCLUDE_DIR_DEBUG
    NAMES gtest/gtest.h
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/debug
    PATH_SUFFIXES include
    DOC "GTest debug include directory"
)

find_library(GTEST_LIBRARY_DEBUG
    NAMES gtest
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/debug
    PATH_SUFFIXES lib lib64
    DOC "GTest debug library"
)

find_library(GTEST_MAIN_LIBRARY_DEBUG
    NAMES gtest_main
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/debug
    PATH_SUFFIXES lib lib64
    DOC "GTest debug main library"
)

find_library(GMOCK_LIBRARY_DEBUG
    NAMES gmock
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/debug
    PATH_SUFFIXES lib lib64
    DOC "GMock debug library"
)

find_library(GMOCK_MAIN_LIBRARY_DEBUG
    NAMES gmock_main
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/debug
    PATH_SUFFIXES lib lib64
    DOC "GMock debug main library"
)

if(GTEST_INCLUDE_DIR_DEBUG AND GTEST_LIBRARY_DEBUG)
    set(_found_debug TRUE)
endif()

# Find release version
find_path(GTEST_INCLUDE_DIR_RELEASE
    NAMES gtest/gtest.h
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/release
    PATH_SUFFIXES include
    DOC "GTest release include directory"
)

find_library(GTEST_LIBRARY_RELEASE
    NAMES gtest
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/release
    PATH_SUFFIXES lib lib64
    DOC "GTest release library"
)

find_library(GTEST_MAIN_LIBRARY_RELEASE
    NAMES gtest_main
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/release
    PATH_SUFFIXES lib lib64
    DOC "GTest release main library"
)

find_library(GMOCK_LIBRARY_RELEASE
    NAMES gmock
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/release
    PATH_SUFFIXES lib lib64
    DOC "GMock release library"
)

find_library(GMOCK_MAIN_LIBRARY_RELEASE
    NAMES gmock_main
    PATHS ${_gtest_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/gtest/release
    PATH_SUFFIXES lib lib64
    DOC "GMock release main library"
)

if(GTEST_INCLUDE_DIR_RELEASE AND GTEST_LIBRARY_RELEASE)
    set(_found_release TRUE)
endif()

# Fallback to old behavior if multi-config versions not found
if(NOT _found_debug AND NOT _found_release)
    find_path(GTEST_INCLUDE_DIR
        NAMES gtest/gtest.h
        PATHS ${_gtest_search_paths}
        PATH_SUFFIXES include
        DOC "GTest include directory"
    )

    find_library(GTEST_LIBRARY
        NAMES gtest
        PATHS ${_gtest_search_paths}
        PATH_SUFFIXES lib lib64
        DOC "GTest library"
    )

    find_library(GTEST_MAIN_LIBRARY
        NAMES gtest_main
        PATHS ${_gtest_search_paths}
        PATH_SUFFIXES lib lib64
        DOC "GTest main library"
    )

    find_library(GMOCK_LIBRARY
        NAMES gmock
        PATHS ${_gtest_search_paths}
        PATH_SUFFIXES lib lib64
        DOC "GMock library"
    )

    find_library(GMOCK_MAIN_LIBRARY
        NAMES gmock_main
        PATHS ${_gtest_search_paths}
        PATH_SUFFIXES lib lib64
        DOC "GMock main library"
    )
endif()

# Determine what we found for the standard args check
if(_found_debug OR _found_release)
    set(GTEST_FOUND TRUE)
    if(_found_debug)
        set(GTEST_INCLUDE_DIR ${GTEST_INCLUDE_DIR_DEBUG})
        set(GTEST_LIBRARY ${GTEST_LIBRARY_DEBUG})
        set(GTEST_MAIN_LIBRARY ${GTEST_MAIN_LIBRARY_DEBUG})
        set(GMOCK_LIBRARY ${GMOCK_LIBRARY_DEBUG})
        set(GMOCK_MAIN_LIBRARY ${GMOCK_MAIN_LIBRARY_DEBUG})
    else()
        set(GTEST_INCLUDE_DIR ${GTEST_INCLUDE_DIR_RELEASE})
        set(GTEST_LIBRARY ${GTEST_LIBRARY_RELEASE})
        set(GTEST_MAIN_LIBRARY ${GTEST_MAIN_LIBRARY_RELEASE})
        set(GMOCK_LIBRARY ${GMOCK_LIBRARY_RELEASE})
        set(GMOCK_MAIN_LIBRARY ${GMOCK_MAIN_LIBRARY_RELEASE})
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GTest
    FOUND_VAR GTEST_FOUND
    REQUIRED_VARS GTEST_INCLUDE_DIR GTEST_LIBRARY
)

# Create imported targets for modern CMake
if(GTEST_FOUND)
    # Main gtest library
    if(NOT TARGET GTest::gtest)
        add_library(GTest::gtest UNKNOWN IMPORTED)
        
        # Use multi-config approach if we found both debug and release
        if(_found_debug AND _found_release)
            # Use generator expressions to select the right version based on config
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION_DEBUG "${GTEST_LIBRARY_DEBUG}"
                IMPORTED_LOCATION_RELEASE "${GTEST_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_RELWITHDEBINFO "${GTEST_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_MINSIZEREL "${GTEST_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "$<IF:$<CONFIG:Debug>,${GTEST_INCLUDE_DIR_DEBUG},${GTEST_INCLUDE_DIR_RELEASE}>"
            )
        elseif(_found_debug)
            # Only debug version found
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION "${GTEST_LIBRARY_DEBUG}"
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_DEBUG}"
            )
        elseif(_found_release)
            # Only release version found
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION "${GTEST_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_RELEASE}"
            )
        else()
            # Fallback to old behavior
            set_target_properties(GTest::gtest PROPERTIES
                IMPORTED_LOCATION "${GTEST_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
            )
        endif()

        # Add pthread on Linux
        if(UNIX AND NOT APPLE)
            find_package(Threads REQUIRED)
            set_property(TARGET GTest::gtest APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES Threads::Threads
            )
        endif()
    endif()

    # gtest_main library
    if(NOT TARGET GTest::gtest_main)
        # Check if we have gtest_main in any configuration
        set(_has_main_lib FALSE)
        if(_found_debug AND _found_release)
            if(GTEST_MAIN_LIBRARY_DEBUG OR GTEST_MAIN_LIBRARY_RELEASE)
                set(_has_main_lib TRUE)
            endif()
        elseif(_found_debug AND GTEST_MAIN_LIBRARY_DEBUG)
            set(_has_main_lib TRUE)
        elseif(_found_release AND GTEST_MAIN_LIBRARY_RELEASE)
            set(_has_main_lib TRUE)
        elseif(GTEST_MAIN_LIBRARY)
            set(_has_main_lib TRUE)
        endif()

        if(_has_main_lib)
            add_library(GTest::gtest_main UNKNOWN IMPORTED)
            
            if(_found_debug AND _found_release)
                # Set properties for both configurations
                if(GTEST_MAIN_LIBRARY_DEBUG)
                    set_target_properties(GTest::gtest_main PROPERTIES
                        IMPORTED_LOCATION_DEBUG "${GTEST_MAIN_LIBRARY_DEBUG}"
                    )
                endif()
                if(GTEST_MAIN_LIBRARY_RELEASE)
                    set_target_properties(GTest::gtest_main PROPERTIES
                        IMPORTED_LOCATION_RELEASE "${GTEST_MAIN_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_RELWITHDEBINFO "${GTEST_MAIN_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_MINSIZEREL "${GTEST_MAIN_LIBRARY_RELEASE}"
                    )
                endif()
                set_target_properties(GTest::gtest_main PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "$<$<CONFIG:Debug>:${GTEST_INCLUDE_DIR_DEBUG}>$<$<NOT:$<CONFIG:Debug>>:${GTEST_INCLUDE_DIR_RELEASE}>"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            elseif(_found_debug)
                set_target_properties(GTest::gtest_main PROPERTIES
                    IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_DEBUG}"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            elseif(_found_release)
                set_target_properties(GTest::gtest_main PROPERTIES
                    IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY_RELEASE}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_RELEASE}"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            else()
                set_target_properties(GTest::gtest_main PROPERTIES
                    IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            endif()
        endif()
    endif()

    # GMock library (optional)
    if(NOT TARGET GTest::gmock)
        # Check if we have gmock in any configuration
        set(_has_gmock_lib FALSE)
        if(_found_debug AND _found_release)
            if(GMOCK_LIBRARY_DEBUG OR GMOCK_LIBRARY_RELEASE)
                set(_has_gmock_lib TRUE)
            endif()
        elseif(_found_debug AND GMOCK_LIBRARY_DEBUG)
            set(_has_gmock_lib TRUE)
        elseif(_found_release AND GMOCK_LIBRARY_RELEASE)
            set(_has_gmock_lib TRUE)
        elseif(GMOCK_LIBRARY)
            set(_has_gmock_lib TRUE)
        endif()

        if(_has_gmock_lib)
            add_library(GTest::gmock UNKNOWN IMPORTED)
            
            if(_found_debug AND _found_release)
                # Set properties for both configurations
                if(GMOCK_LIBRARY_DEBUG)
                    set_target_properties(GTest::gmock PROPERTIES
                        IMPORTED_LOCATION_DEBUG "${GMOCK_LIBRARY_DEBUG}"
                    )
                endif()
                if(GMOCK_LIBRARY_RELEASE)
                    set_target_properties(GTest::gmock PROPERTIES
                        IMPORTED_LOCATION_RELEASE "${GMOCK_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_RELWITHDEBINFO "${GMOCK_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_MINSIZEREL "${GMOCK_LIBRARY_RELEASE}"
                    )
                endif()
                set_target_properties(GTest::gmock PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "$<$<CONFIG:Debug>:${GTEST_INCLUDE_DIR_DEBUG}>$<$<NOT:$<CONFIG:Debug>>:${GTEST_INCLUDE_DIR_RELEASE}>"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            elseif(_found_debug)
                set_target_properties(GTest::gmock PROPERTIES
                    IMPORTED_LOCATION "${GMOCK_LIBRARY_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_DEBUG}"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            elseif(_found_release)
                set_target_properties(GTest::gmock PROPERTIES
                    IMPORTED_LOCATION "${GMOCK_LIBRARY_RELEASE}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_RELEASE}"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            else()
                set_target_properties(GTest::gmock PROPERTIES
                    IMPORTED_LOCATION "${GMOCK_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                    INTERFACE_LINK_LIBRARIES "GTest::gtest"
                )
            endif()
        endif()
    endif()

    # GMock main library (optional)
    if(NOT TARGET GTest::gmock_main)
        # Check if we have gmock_main in any configuration
        set(_has_gmock_main_lib FALSE)
        if(_found_debug AND _found_release)
            if(GMOCK_MAIN_LIBRARY_DEBUG OR GMOCK_MAIN_LIBRARY_RELEASE)
                set(_has_gmock_main_lib TRUE)
            endif()
        elseif(_found_debug AND GMOCK_MAIN_LIBRARY_DEBUG)
            set(_has_gmock_main_lib TRUE)
        elseif(_found_release AND GMOCK_MAIN_LIBRARY_RELEASE)
            set(_has_gmock_main_lib TRUE)
        elseif(GMOCK_MAIN_LIBRARY)
            set(_has_gmock_main_lib TRUE)
        endif()

        if(_has_gmock_main_lib)
            add_library(GTest::gmock_main UNKNOWN IMPORTED)
            
            if(_found_debug AND _found_release)
                # Set properties for both configurations
                if(GMOCK_MAIN_LIBRARY_DEBUG)
                    set_target_properties(GTest::gmock_main PROPERTIES
                        IMPORTED_LOCATION_DEBUG "${GMOCK_MAIN_LIBRARY_DEBUG}"
                    )
                endif()
                if(GMOCK_MAIN_LIBRARY_RELEASE)
                    set_target_properties(GTest::gmock_main PROPERTIES
                        IMPORTED_LOCATION_RELEASE "${GMOCK_MAIN_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_RELWITHDEBINFO "${GMOCK_MAIN_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_MINSIZEREL "${GMOCK_MAIN_LIBRARY_RELEASE}"
                    )
                endif()
                set_target_properties(GTest::gmock_main PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "$<$<CONFIG:Debug>:${GTEST_INCLUDE_DIR_DEBUG}>$<$<NOT:$<CONFIG:Debug>>:${GTEST_INCLUDE_DIR_RELEASE}>"
                    INTERFACE_LINK_LIBRARIES "GTest::gmock"
                )
            elseif(_found_debug)
                set_target_properties(GTest::gmock_main PROPERTIES
                    IMPORTED_LOCATION "${GMOCK_MAIN_LIBRARY_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_DEBUG}"
                    INTERFACE_LINK_LIBRARIES "GTest::gmock"
                )
            elseif(_found_release)
                set_target_properties(GTest::gmock_main PROPERTIES
                    IMPORTED_LOCATION "${GMOCK_MAIN_LIBRARY_RELEASE}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR_RELEASE}"
                    INTERFACE_LINK_LIBRARIES "GTest::gmock"
                )
            else()
                set_target_properties(GTest::gmock_main PROPERTIES
                    IMPORTED_LOCATION "${GMOCK_MAIN_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}"
                    INTERFACE_LINK_LIBRARIES "GTest::gmock"
                )
            endif()
        endif()
    endif()

    # Enable gtest_discover_tests() if available
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.10")
        include(GoogleTest OPTIONAL)
    endif()
endif()

# Clean up temporary variables
unset(_gtest_search_paths)
unset(_found_debug)
unset(_found_release)
unset(_has_main_lib)
unset(_has_gmock_lib)
unset(_has_gmock_main_lib)
unset(GTEST_INCLUDE_DIR)
unset(GTEST_LIBRARY)
unset(GTEST_MAIN_LIBRARY)
unset(GMOCK_LIBRARY)
unset(GMOCK_MAIN_LIBRARY)
unset(GTEST_INCLUDE_DIR_DEBUG)
unset(GTEST_LIBRARY_DEBUG)
unset(GTEST_MAIN_LIBRARY_DEBUG)
unset(GMOCK_LIBRARY_DEBUG)
unset(GMOCK_MAIN_LIBRARY_DEBUG)
unset(GTEST_INCLUDE_DIR_RELEASE)
unset(GTEST_LIBRARY_RELEASE)
unset(GTEST_MAIN_LIBRARY_RELEASE)
unset(GMOCK_LIBRARY_RELEASE)
unset(GMOCK_MAIN_LIBRARY_RELEASE)