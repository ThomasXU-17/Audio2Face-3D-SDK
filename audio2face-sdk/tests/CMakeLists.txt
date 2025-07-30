# Define the test executable target
add_executable(audio2face-unit-tests)

set_target_properties(audio2face-unit-tests PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/bin
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TENSORRT_ROOT_DIR}/lib;${CUDA_PATH}/bin"
    VS_DEBUGGER_WORKING_DIRECTORY ${AUDIO2X_SDK_ROOT}
)

# Add all test source files
file(GLOB_RECURSE TEST_SOURCES
    "*.cpp"
    "*.cu"
)

target_sources(audio2face-unit-tests PRIVATE ${TEST_SOURCES})

# Include directories
target_include_directories(audio2face-unit-tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../audio2x-common/include
)

# Set compile definitions
target_compile_definitions(audio2face-unit-tests PRIVATE
    AUDIO2X_SDK_EXPORTS
    AUDIO2FACE_SDK_DLL_EXPORTS
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

# Link with required libraries
target_link_libraries(audio2face-unit-tests
    PRIVATE
        audio2face-core
        audio2x-common
        Eigen3::Eigen
        refl-cpp::refl-cpp
        tbtSVD::tbtSVD
        GTest::gtest
        GTest::gtest_main
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    target_link_libraries(audio2face-unit-tests PRIVATE pthread)
endif()

gtest_discover_tests(audio2face-unit-tests DISCOVERY_MODE PRE_TEST)
