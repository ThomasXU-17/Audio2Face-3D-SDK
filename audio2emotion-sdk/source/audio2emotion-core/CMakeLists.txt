# Define the static library target
add_library(audio2emotion-core STATIC)

set_target_properties(audio2emotion-core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/bin
)

# Add all source files from source/audio2emotion-core
file(GLOB_RECURSE AUDIO2EMOTION_CORE_SOURCES
    "*.cpp"
    "*.cu"
)

target_sources(audio2emotion-core PRIVATE ${AUDIO2EMOTION_CORE_SOURCES})

# Include directories
target_include_directories(audio2emotion-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/audio2emotion/internal
)

# Add compile definitions
target_compile_definitions(audio2emotion-core
    PRIVATE
        AUDIO2X_SDK_EXPORTS
        $<$<PLATFORM_ID:Windows>:AUDIO2EMOTION_SDK_DLL_EXPORTS>
)

# Link libraries
target_link_libraries(audio2emotion-core
    PUBLIC
        audio2x-common
    PRIVATE
        nlohmann_json::nlohmann_json
)
