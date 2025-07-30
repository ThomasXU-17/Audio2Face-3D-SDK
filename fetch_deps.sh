#!/bin/bash

# Get the directory where this script is located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_CONFIG=release
# Check if a build configuration was provided as an argument
if [ "$1" != "" ]; then
    BUILD_CONFIG="$1"
fi

if [ "$PYTHON" = "" ]; then
    PYTHON="python"
fi

if [ "$PACKMAN" = "" ]; then
    PACKMAN="${BASE_DIR}/tools/packman/packman"
fi

# Pull dependencies using packman
"$PACKMAN" pull -t config="$BUILD_CONFIG" --platform linux-x86_64 "${BASE_DIR}/deps/build-deps.packman.xml"
if [ $? -ne 0 ]; then
    echo "Failed to pull dependencies in build-deps.packman.xml"
    exit 1
fi

"$PACKMAN" pull -t config="$BUILD_CONFIG" --platform linux-x86_64 "${BASE_DIR}/deps/target-deps.packman.xml"
if [ $? -ne 0 ]; then
    echo "Failed to pull dependencies in target-deps.packman.xml"
    exit 1
fi