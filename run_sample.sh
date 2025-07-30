#!/bin/bash

# Set the base directories.
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$BASE_DIR/_build"
TARGET_DEPS="$BASE_DIR/_deps/target-deps"

# Detect release or debug mode from the first argument, remove it from the command.
if [[ "$1" == "debug" ]]; then
    BUILD_TYPE="debug"
    shift
    DEBUG_DIR="${BUILD_DIR}/debug/"
    if [[ ! -d "$DEBUG_DIR" ]]; then
        echo "Debug build directory does not exist: $DEBUG_DIR"
        echo "Please run \`build.sh all debug\` first."
        exit 1
    fi
elif [[ "$1" == "release" ]]; then
    BUILD_TYPE="release"
    shift
    RELEASE_DIR="${BUILD_DIR}/release/"
    if [[ ! -d "$RELEASE_DIR" ]]; then
        echo "Release build directory does not exist: $RELEASE_DIR"
        echo "Please run \`build.sh all release\` first."
        exit 1
    fi
fi

# If BUILD_TYPE is not set, check if only one of the build directories exists and set BUILD_TYPE accordingly.
if [[ -z "$BUILD_TYPE" ]]; then
    RELEASE_DIR="${BUILD_DIR}/release/"
    DEBUG_DIR="${BUILD_DIR}/debug/"
    if [[ -d "$RELEASE_DIR" && ! -d "$DEBUG_DIR" ]]; then
        BUILD_TYPE="release"
    elif [[ -d "$DEBUG_DIR" && ! -d "$RELEASE_DIR" ]]; then
        BUILD_TYPE="debug"
    elif [[ -d "$RELEASE_DIR" && -d "$DEBUG_DIR" ]]; then
        # Default to release if both build directories exist and no argument is provided.
        BUILD_TYPE="release"
    else
        echo "No build directory exists. Please run \`build.sh all\` first."
        exit 1
    fi
fi

export PATH="${BUILD_DIR}/${BUILD_TYPE}/audio2x-sdk/bin:${PATH}"
export PYTHONPATH="${BASE_DIR}/audio2x-common/scripts:${PYTHONPATH}"

# Add CUDA bin dir to PATH if CUDA_PATH is defined
if [ -n "${CUDA_PATH}" ]; then
    export PATH="${CUDA_PATH}/bin:${PATH}"
else
    echo CUDA_PATH is not defined
    exit 1
fi

# Add TensorRT lib dir to PATH if TENSORRT_ROOT_DIR is defined
if [ -n "${TENSORRT_ROOT_DIR}" ]; then
    export PATH="${TENSORRT_ROOT_DIR}/lib:${PATH}"
else
    echo TENSORRT_ROOT_DIR is not defined
    exit 1
fi

# If the first argument is not a file, try to find the corresponding executable in the build directory.
if [ ! -f "$1" ]; then
    RELATIVE_PATH="${BUILD_DIR}/${BUILD_TYPE}/"
    if [ -f "$RELATIVE_PATH/$1" ]; then
        exec "$RELATIVE_PATH/$1" "${@:2}"
    else
        echo "Error: Found neither an absolute path:"
        echo "  $1" 
        echo "nor a relative path to the build directory:"
        echo "  $ALT_CMD"
        exit 1
    fi
fi

exec "$@"
