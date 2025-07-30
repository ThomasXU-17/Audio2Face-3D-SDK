#!/bin/bash

set -e  # Exit on any error

# Set the default build configuration to release
BUILD_CONFIG="release"
# Set the default build project to all
BUILD_PROJECT="all"

# Select the build configuration and the project to build in arbitrary order.
if [ "$1" = "release" ] || [ "$1" = "debug" ]; then
    BUILD_CONFIG="$1"
    if [ -n "$2" ]; then
        BUILD_PROJECT="$2"
    fi
else
    BUILD_PROJECT="$1"
    if [ -n "$2" ]; then
        BUILD_CONFIG="$2"
    fi
fi

BASE_DIR="$(dirname ${BASH_SOURCE})"
BUILD_DIR="${BASE_DIR}/_build/${BUILD_CONFIG}"
export PATH="$PATH:${BASE_DIR}/_deps/build-deps/ninja"

CMAKE="${BASE_DIR}/_deps/build-deps/cmake/bin/cmake"

# Set the default build configuration to release
BUILD_CONFIG="release"

# Check if a build configuration was provided as an argument
if [ -n "$2" ]; then
    BUILD_CONFIG="$2"
fi

BUILD_DIR="${BASE_DIR}/_build/${BUILD_CONFIG}"

BUILD_PROJECT="all"
if [ "$1" = "clean" ]; then
    rm -rf "$BUILD_DIR"
    exit 0
fi

if [ ! -d "${BASE_DIR}/_deps" ]; then
    echo "Dependencies not found. Please run ./fetch_deps.sh ${BUILD_CONFIG} first."
    exit 1
fi

if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

# Check if cmake exists
if [ ! -f "$CMAKE" ]; then
    echo "CMake not found at $CMAKE. Please ensure dependencies are fetched."
    exit 1
fi

# Check if ninja exists
if ! command -v ninja >/dev/null 2>&1; then
    echo "Ninja not found in PATH. Please ensure dependencies are fetched."
    exit 1
fi

"$CMAKE" -B "$BUILD_DIR" -G Ninja -S . -DCMAKE_BUILD_TYPE="${BUILD_CONFIG^}"

"$CMAKE" --build "$BUILD_DIR" --target "$BUILD_PROJECT" --config "$BUILD_CONFIG" --parallel
