add_executable(sample-a2e-inference)

set_target_properties(sample-a2e-inference PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2E_BINARY_DIR}/bin
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TENSORRT_ROOT_DIR}/lib;${CUDA_PATH}/bin;$<TARGET_FILE_DIR:audio2x>"
    VS_DEBUGGER_WORKING_DIRECTORY ${AUDIO2X_SDK_ROOT}
)

target_sources(sample-a2e-inference PRIVATE
    main.cpp
)

target_compile_definitions(sample-a2e-inference PRIVATE
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

target_link_libraries(sample-a2e-inference PRIVATE
    audio2x
    AudioFile::AudioFile
)
