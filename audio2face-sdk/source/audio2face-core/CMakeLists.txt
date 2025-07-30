# Define the static library target
add_library(audio2face-core STATIC)

# Add all source files from source/audio2face-core
file(GLOB_RECURSE AUDIO2FACE_CORE_SOURCES
    "*.cpp"
    "*.cu"
)

target_sources(audio2face-core PRIVATE ${AUDIO2FACE_CORE_SOURCES})

# Include directories
target_include_directories(audio2face-core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include/audio2face/internal
)

# Add compile definitions
target_compile_definitions(audio2face-core
    PRIVATE
        AUDIO2X_SDK_EXPORTS
        $<$<PLATFORM_ID:Windows>:AUDIO2FACE_SDK_DLL_EXPORTS>
)

# Link libraries
target_link_libraries(audio2face-core
    PUBLIC
        CUDA::cublas
        CUDA::curand
        audio2x-common
    PRIVATE
        Eigen3::Eigen
        magic_enum::magic_enum
        nlohmann_json::nlohmann_json
        refl-cpp::refl-cpp
        tbtSVD::tbtSVD
)
