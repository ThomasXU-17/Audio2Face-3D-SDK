add_executable(a2f-web-inference)

set_target_properties(a2f-web-inference PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/bin
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TENSORRT_ROOT_DIR}/lib;${CUDA_PATH}/bin;$<TARGET_FILE_DIR:audio2x>"
    VS_DEBUGGER_WORKING_DIRECTORY ${AUDIO2X_SDK_ROOT}
)

target_sources(a2f-web-inference PRIVATE
    main.cpp
)

target_compile_definitions(a2f-web-inference PRIVATE
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

target_link_libraries(a2f-web-inference PRIVATE
    audio2x
    AudioFile::AudioFile
    CXXOPTS::CXXOPTS
    nlohmann_json::nlohmann_json
    CUDA::cudart
)
