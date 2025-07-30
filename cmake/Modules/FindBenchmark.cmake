# FindBenchmark.cmake
# Find the Google Benchmark library
#
# This module creates the following imported targets:
#  Benchmark::Benchmark      - The main Benchmark library target
#  Benchmark::Benchmark_main - The Benchmark main library target (optional)
#
# Usage:
#  find_package(Benchmark REQUIRED)
#  target_link_libraries(your_target PRIVATE Benchmark::Benchmark)
#
# You can specify the Benchmark installation directory by setting:
#  BENCHMARK_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_benchmark_search_paths)

# Priority 1: CMake variable BENCHMARK_ROOT
if(BENCHMARK_ROOT)
    list(APPEND _benchmark_search_paths ${BENCHMARK_ROOT})
endif()

# Priority 2: Environment variable BENCHMARK_ROOT
if(DEFINED ENV{BENCHMARK_ROOT})
    list(APPEND _benchmark_search_paths $ENV{BENCHMARK_ROOT})
endif()

# Priority 3: Project's dependency location
list(APPEND _benchmark_search_paths
    ${CMAKE_SOURCE_DIR}/_build/target-deps/benchmark
)

# Priority 4: Standard system locations
list(APPEND _benchmark_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# For multi-config generators, try to find both debug and release versions
set(_found_debug FALSE)
set(_found_release FALSE)

# Find debug version
find_path(BENCHMARK_INCLUDE_DIR_DEBUG
    NAMES benchmark/benchmark.h
    PATHS ${_benchmark_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/benchmark/debug
    PATH_SUFFIXES include
    DOC "Benchmark debug include directory"
)

find_library(BENCHMARK_LIBRARY_DEBUG
    NAMES benchmark
    PATHS ${_benchmark_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/benchmark/debug
    PATH_SUFFIXES lib lib64
    DOC "Benchmark debug library"
)

find_library(BENCHMARK_MAIN_LIBRARY_DEBUG
    NAMES benchmark_main
    PATHS ${_benchmark_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/benchmark/debug
    PATH_SUFFIXES lib lib64
    DOC "Benchmark debug main library"
)

if(BENCHMARK_INCLUDE_DIR_DEBUG AND BENCHMARK_LIBRARY_DEBUG)
    set(_found_debug TRUE)
endif()

# Find release version
find_path(BENCHMARK_INCLUDE_DIR_RELEASE
    NAMES benchmark/benchmark.h
    PATHS ${_benchmark_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/benchmark/release
    PATH_SUFFIXES include
    DOC "Benchmark release include directory"
)

find_library(BENCHMARK_LIBRARY_RELEASE
    NAMES benchmark
    PATHS ${_benchmark_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/benchmark/release
    PATH_SUFFIXES lib lib64
    DOC "Benchmark release library"
)

find_library(BENCHMARK_MAIN_LIBRARY_RELEASE
    NAMES benchmark_main
    PATHS ${_benchmark_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/benchmark/release
    PATH_SUFFIXES lib lib64
    DOC "Benchmark release main library"
)

if(BENCHMARK_INCLUDE_DIR_RELEASE AND BENCHMARK_LIBRARY_RELEASE)
    set(_found_release TRUE)
endif()

# Fallback to old behavior if multi-config versions not found
if(NOT _found_debug AND NOT _found_release)
    find_path(BENCHMARK_INCLUDE_DIR
        NAMES benchmark/benchmark.h
        PATHS ${_benchmark_search_paths}
        PATH_SUFFIXES include
        DOC "Benchmark include directory"
    )

    find_library(BENCHMARK_LIBRARY
        NAMES benchmark
        PATHS ${_benchmark_search_paths}
        PATH_SUFFIXES lib lib64
        DOC "Benchmark library"
    )

    find_library(BENCHMARK_MAIN_LIBRARY
        NAMES benchmark_main
        PATHS ${_benchmark_search_paths}
        PATH_SUFFIXES lib lib64
        DOC "Benchmark main library"
    )
endif()

# Determine what we found for the standard args check
if(_found_debug OR _found_release)
    set(BENCHMARK_FOUND TRUE)
    if(_found_debug)
        set(BENCHMARK_INCLUDE_DIR ${BENCHMARK_INCLUDE_DIR_DEBUG})
        set(BENCHMARK_LIBRARY ${BENCHMARK_LIBRARY_DEBUG})
        set(BENCHMARK_MAIN_LIBRARY ${BENCHMARK_MAIN_LIBRARY_DEBUG})
    else()
        set(BENCHMARK_INCLUDE_DIR ${BENCHMARK_INCLUDE_DIR_RELEASE})
        set(BENCHMARK_LIBRARY ${BENCHMARK_LIBRARY_RELEASE})
        set(BENCHMARK_MAIN_LIBRARY ${BENCHMARK_MAIN_LIBRARY_RELEASE})
    endif()
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Benchmark
    FOUND_VAR BENCHMARK_FOUND
    REQUIRED_VARS BENCHMARK_INCLUDE_DIR BENCHMARK_LIBRARY
)

# Create imported targets for modern CMake
if(BENCHMARK_FOUND)
    # Main benchmark library
    if(NOT TARGET Benchmark::Benchmark)
        add_library(Benchmark::Benchmark UNKNOWN IMPORTED)
        
        # Use multi-config approach if we found both debug and release
        if(_found_debug AND _found_release)
            # Use generator expressions to select the right version based on config
            set_target_properties(Benchmark::Benchmark PROPERTIES
                IMPORTED_LOCATION_DEBUG "${BENCHMARK_LIBRARY_DEBUG}"
                IMPORTED_LOCATION_RELEASE "${BENCHMARK_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_MINSIZEREL "${BENCHMARK_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_RELWITHDEBINFO "${BENCHMARK_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "$<IF:$<CONFIG:Debug>,${BENCHMARK_INCLUDE_DIR_DEBUG},${BENCHMARK_INCLUDE_DIR_RELEASE}>"
                INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
            )
        elseif(_found_debug)
            # Only debug version found
            set_target_properties(Benchmark::Benchmark PROPERTIES
                IMPORTED_LOCATION "${BENCHMARK_LIBRARY_DEBUG}"
                INTERFACE_INCLUDE_DIRECTORIES "${BENCHMARK_INCLUDE_DIR_DEBUG}"
                INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
            )
        elseif(_found_release)
            # Only release version found
            set_target_properties(Benchmark::Benchmark PROPERTIES
                IMPORTED_LOCATION "${BENCHMARK_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "${BENCHMARK_INCLUDE_DIR_RELEASE}"
                INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
            )
        else()
            # Fallback to old behavior
            set_target_properties(Benchmark::Benchmark PROPERTIES
                IMPORTED_LOCATION "${BENCHMARK_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${BENCHMARK_INCLUDE_DIR}"
                INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
            )
        endif()

        # Add Windows system libraries that benchmark depends on
        if(WIN32)
            set_property(TARGET Benchmark::Benchmark APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES "shlwapi"
            )
        endif()

        # Add pthread on Linux
        if(UNIX AND NOT APPLE)
            find_package(Threads REQUIRED)
            set_property(TARGET Benchmark::Benchmark APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES Threads::Threads
            )
        endif()
    endif()

    # Optional benchmark_main library
    if(NOT TARGET Benchmark::Benchmark_main)
        # Check if we have benchmark_main in any configuration
        set(_has_main_lib FALSE)
        if(_found_debug AND _found_release)
            if(BENCHMARK_MAIN_LIBRARY_DEBUG OR BENCHMARK_MAIN_LIBRARY_RELEASE)
                set(_has_main_lib TRUE)
            endif()
        elseif(_found_debug AND BENCHMARK_MAIN_LIBRARY_DEBUG)
            set(_has_main_lib TRUE)
        elseif(_found_release AND BENCHMARK_MAIN_LIBRARY_RELEASE)
            set(_has_main_lib TRUE)
        elseif(BENCHMARK_MAIN_LIBRARY)
            set(_has_main_lib TRUE)
        endif()

        if(_has_main_lib)
            add_library(Benchmark::Benchmark_main UNKNOWN IMPORTED)
            
            if(_found_debug AND _found_release)
                # Set properties for both configurations
                if(BENCHMARK_MAIN_LIBRARY_DEBUG)
                    set_target_properties(Benchmark::Benchmark_main PROPERTIES
                        IMPORTED_LOCATION_DEBUG "${BENCHMARK_MAIN_LIBRARY_DEBUG}"
                    )
                endif()
                if(BENCHMARK_MAIN_LIBRARY_RELEASE)
                    set_target_properties(Benchmark::Benchmark_main PROPERTIES
                        IMPORTED_LOCATION_RELEASE "${BENCHMARK_MAIN_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_RELWITHDEBINFO "${BENCHMARK_MAIN_LIBRARY_RELEASE}"
                        IMPORTED_LOCATION_MINSIZEREL "${BENCHMARK_LIBRARY_RELEASE}"
                    )
                endif()
                set_target_properties(Benchmark::Benchmark_main PROPERTIES
                    INTERFACE_INCLUDE_DIRECTORIES "$<$<CONFIG:Debug>:${BENCHMARK_INCLUDE_DIR_DEBUG}>$<$<NOT:$<CONFIG:Debug>>:${BENCHMARK_INCLUDE_DIR_RELEASE}>"
                    INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
                )
            elseif(_found_debug)
                set_target_properties(Benchmark::Benchmark_main PROPERTIES
                    IMPORTED_LOCATION "${BENCHMARK_MAIN_LIBRARY_DEBUG}"
                    INTERFACE_INCLUDE_DIRECTORIES "${BENCHMARK_INCLUDE_DIR_DEBUG}"
                    INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
                )
            elseif(_found_release)
                set_target_properties(Benchmark::Benchmark_main PROPERTIES
                    IMPORTED_LOCATION "${BENCHMARK_MAIN_LIBRARY_RELEASE}"
                    INTERFACE_INCLUDE_DIRECTORIES "${BENCHMARK_INCLUDE_DIR_RELEASE}"
                    INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
                )
            else()
                set_target_properties(Benchmark::Benchmark_main PROPERTIES
                    IMPORTED_LOCATION "${BENCHMARK_MAIN_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${BENCHMARK_INCLUDE_DIR}"
                    INTERFACE_COMPILE_DEFINITIONS "BENCHMARK_STATIC_DEFINE"
                )
            endif()
        endif()
    endif()
endif()

# Clean up temporary variables
unset(_benchmark_search_paths)
unset(_found_debug)
unset(_found_release)
unset(_has_main_lib)
unset(BENCHMARK_INCLUDE_DIR)
unset(BENCHMARK_LIBRARY)
unset(BENCHMARK_MAIN_LIBRARY)
unset(BENCHMARK_INCLUDE_DIR_DEBUG)
unset(BENCHMARK_LIBRARY_DEBUG)
unset(BENCHMARK_MAIN_LIBRARY_DEBUG)
unset(BENCHMARK_INCLUDE_DIR_RELEASE)
unset(BENCHMARK_LIBRARY_RELEASE)
unset(BENCHMARK_MAIN_LIBRARY_RELEASE)