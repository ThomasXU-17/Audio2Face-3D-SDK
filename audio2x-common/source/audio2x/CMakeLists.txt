# Define the static library target
add_library(audio2x-common STATIC)

set_target_properties(audio2x-common PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2X_COMMON_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2X_COMMON_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2X_COMMON_BINARY_DIR}/bin
)

# Add all source files from source/audio2x
file(GLOB_RECURSE AUDIO2X_SOURCES
    "*.cpp"
    "*.cu"
)

target_sources(audio2x-common PRIVATE ${AUDIO2X_SOURCES})

# Include directories
target_include_directories(audio2x-common
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/audio2x/internal
)

# Add compile definitions
target_compile_definitions(audio2x-common PRIVATE AUDIO2X_SDK_EXPORTS)

target_link_libraries(audio2x-common
    PUBLIC
        Cnpy::Cnpy
        CUDA::cudart
        TensorRT::TensorRT
        ZLIB::ZLIB
    PRIVATE
        AudioFile::AudioFile
)
