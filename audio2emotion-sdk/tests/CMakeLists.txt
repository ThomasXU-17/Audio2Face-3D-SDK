# Define the test executable target
add_executable(audio2emotion-unit-tests)

set_target_properties(audio2emotion-unit-tests PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/bin
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TENSORRT_ROOT_DIR}/lib;${CUDA_PATH}/bin"
    VS_DEBUGGER_WORKING_DIRECTORY ${AUDIO2X_SDK_ROOT}
)   

# Add all test source files
file(GLOB_RECURSE TEST_SOURCES
    "*.cpp"
    "*.cu"
)

target_sources(audio2emotion-unit-tests PRIVATE ${TEST_SOURCES})

# Include directories
target_include_directories(audio2emotion-unit-tests
    PRIVATE
        # Hardcoded includes based on premake5.lua - to be converted to Find modules later
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../audio2x-common/include
)

# Set compile definitions
target_compile_definitions(audio2emotion-unit-tests PRIVATE
    AUDIO2X_SDK_EXPORTS
    $<$<PLATFORM_ID:Windows>:AUDIO2EMOTION_SDK_DLL_EXPORTS>
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

# Link with required libraries
target_link_libraries(audio2emotion-unit-tests
    PRIVATE
        audio2emotion-core
        audio2x-common
        GTest::gtest
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
        CUDA::curand
        CUDA::nvrtc
        TensorRT::TensorRT
        ZLIB::ZLIB
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    target_link_libraries(audio2emotion-unit-tests PRIVATE pthread)
endif()

gtest_discover_tests(audio2emotion-unit-tests DISCOVERY_MODE PRE_TEST)
