# Define the test executable target
add_executable(audio2x-common-unit-tests)

set_target_properties(audio2x-common-unit-tests PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2X_COMMON_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2X_COMMON_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2X_COMMON_BINARY_DIR}/bin
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TENSORRT_ROOT_DIR}/lib;${CUDA_PATH}/bin"
    VS_DEBUGGER_WORKING_DIRECTORY ${AUDIO2X_SDK_ROOT}
)

# Add all test source files
file(GLOB_RECURSE TEST_SOURCES
    "*.cpp"
)

target_sources(audio2x-common-unit-tests PRIVATE ${TEST_SOURCES})

# Set compile definitions
target_compile_definitions(audio2x-common-unit-tests PRIVATE
    AUDIO2X_SDK_STATIC_DEFINE
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

# Link with required libraries
target_link_libraries(audio2x-common-unit-tests
    PRIVATE
        audio2x-common
        GTest::gtest
        GTest::gtest_main
)

# Platform-specific linking
if(UNIX AND NOT APPLE)
    target_link_libraries(audio2x-common-unit-tests PRIVATE pthread)
endif()

gtest_discover_tests(audio2x-common-unit-tests DISCOVERY_MODE PRE_TEST)
