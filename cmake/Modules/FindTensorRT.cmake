# FindTensorRT.cmake
# Find the TensorRT library
#
# This module creates the following imported target:
#  TensorRT::TensorRT   - The TensorRT library target
#
# Usage:
#  find_package(TensorRT REQUIRED)
#  target_link_libraries(your_target PRIVATE TensorRT::TensorRT)
#
# You can specify the TensorRT installation directory by setting:
#  TENSORRT_ROOT_DIR         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.
#
# This module also provides the following variables for compatibility:
#  TENSORRT_FOUND        - True if TensorRT is found
#  TENSORRT_VERSION      - Version of TensorRT (e.g., "10.0.1")

# Allow overriding the search path via environment variable or CMake variable
set(_tensorrt_search_paths)

# Priority 1: CMake variable TensorRT_ROOT
if(TENSORRT_ROOT_DIR)
    list(APPEND _tensorrt_search_paths ${TENSORRT_ROOT_DIR})
endif()

# Priority 2: Environment variable TensorRT_ROOT
if(DEFINED ENV{TENSORRT_ROOT_DIR})
    list(APPEND _tensorrt_search_paths $ENV{TENSORRT_ROOT_DIR})
endif()

# Priority 3: Standard system locations
list(APPEND _tensorrt_search_paths
    /usr/local/cuda
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${_tensorrt_search_paths}
    PATH_SUFFIXES include cuda/include
    DOC "TensorRT include directory"
)

# Find required libraries
find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer nvinfer_10
    PATHS ${_tensorrt_search_paths}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64
    DOC "TensorRT nvinfer library"
)

find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY
    NAMES nvinfer_plugin nvinfer_plugin_10
    PATHS ${_tensorrt_search_paths}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64
    DOC "TensorRT nvinfer_plugin library"
)

find_library(TENSORRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser nvonnxparser_10
    PATHS ${_tensorrt_search_paths}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64
    DOC "TensorRT nvonnxparser library"
)

# Extract version information if header is found
if(TENSORRT_INCLUDE_DIR AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(READ ${TENSORRT_INCLUDE_DIR}/NvInferVersion.h TENSORRT_HEADER_CONTENTS)
    string(REGEX MATCH "define NV_TENSORRT_MAJOR * +([0-9]+)"
                 TENSORRT_VERSION_MAJOR "${TENSORRT_HEADER_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_MAJOR * +([0-9]+)" "\\1"
                 TENSORRT_VERSION_MAJOR "${TENSORRT_VERSION_MAJOR}")
    string(REGEX MATCH "define NV_TENSORRT_MINOR * +([0-9]+)"
                 TENSORRT_VERSION_MINOR "${TENSORRT_HEADER_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_MINOR * +([0-9]+)" "\\1"
                 TENSORRT_VERSION_MINOR "${TENSORRT_VERSION_MINOR}")
    string(REGEX MATCH "define NV_TENSORRT_PATCH * +([0-9]+)"
                 TENSORRT_VERSION_PATCH "${TENSORRT_HEADER_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_PATCH * +([0-9]+)" "\\1"
                 TENSORRT_VERSION_PATCH "${TENSORRT_VERSION_PATCH}")
    string(REGEX MATCH "define NV_TENSORRT_BUILD * +([0-9]+)"
                 TENSORRT_VERSION_BUILD "${TENSORRT_HEADER_CONTENTS}")
    string(REGEX REPLACE "define NV_TENSORRT_BUILD * +([0-9]+)" "\\1"
                 TENSORRT_VERSION_BUILD "${TENSORRT_VERSION_BUILD}")

    # Assemble TensorRT version
    if(NOT TENSORRT_VERSION_MAJOR)
        set(TENSORRT_VERSION "?")
    else()
        set(TENSORRT_VERSION
            "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}.${TENSORRT_VERSION_BUILD}")
    endif()
else()
    set(TENSORRT_VERSION "unknown")
endif()

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    FOUND_VAR TENSORRT_FOUND
    REQUIRED_VARS
        TENSORRT_INCLUDE_DIR
        TENSORRT_NVINFER_LIBRARY
        TENSORRT_NVINFER_PLUGIN_LIBRARY
        TENSORRT_NVONNXPARSER_LIBRARY
    VERSION_VAR TENSORRT_VERSION
)

# Create imported target for modern CMake
if(TENSORRT_FOUND)
    if(NOT TARGET TensorRT::TensorRT)
        add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
        set_target_properties(TensorRT::TensorRT PROPERTIES
            IMPORTED_LOCATION "${TENSORRT_NVINFER_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TENSORRT_INCLUDE_DIR}"
            INTERFACE_LINK_LIBRARIES "${TENSORRT_NVINFER_PLUGIN_LIBRARY};${TENSORRT_NVONNXPARSER_LIBRARY}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_tensorrt_search_paths)
unset(TENSORRT_HEADER_CONTENTS)
unset(TENSORRT_INCLUDE_DIR)
unset(TENSORRT_NVINFER_LIBRARY)
unset(TENSORRT_NVINFER_PLUGIN_LIBRARY)
unset(TENSORRT_NVONNXPARSER_LIBRARY)
unset(TENSORRT_VERSION_MAJOR)
unset(TENSORRT_VERSION_MINOR)
unset(TENSORRT_VERSION_PATCH)
