# Single benchmark executable that includes all .cpp files
add_executable(audio2face-benchmarks)

set_target_properties(audio2face-benchmarks PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${A2F_BINARY_DIR}/bin
    VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TENSORRT_ROOT_DIR}/lib;${CUDA_PATH}/bin;$<TARGET_FILE_DIR:audio2x>"
    VS_DEBUGGER_WORKING_DIRECTORY ${AUDIO2X_SDK_ROOT}
)

target_sources(audio2face-benchmarks PRIVATE
    main.cpp
    blendshape_solver.cpp
    executor_blendshape.cpp
    executor_geometry.cpp
    interactive_executor.cpp
    utils.cpp
)

target_compile_definitions(audio2face-benchmarks PRIVATE
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

target_link_libraries(audio2face-benchmarks PRIVATE
    audio2x
    AudioFile::AudioFile
    Benchmark::Benchmark
    Cnpy::Cnpy
    ZLIB::ZLIB
)
