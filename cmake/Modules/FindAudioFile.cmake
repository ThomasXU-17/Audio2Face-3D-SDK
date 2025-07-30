# FindAudioFile.cmake
# Find the AudioFile header-only library
#
# This module creates the following imported target:
#  AudioFile::AudioFile   - The AudioFile header-only library target
#
# Usage:
#  find_package(AudioFile REQUIRED)
#  target_link_libraries(your_target PRIVATE AudioFile::AudioFile)
#
# You can specify the AudioFile installation directory by setting:
#  AUDIOFILE_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_audiofile_search_paths)

# Priority 1: CMake variable AUDIOFILE_ROOT
if(AUDIOFILE_ROOT)
    list(APPEND _audiofile_search_paths ${AUDIOFILE_ROOT})
endif()

# Priority 2: Environment variable AUDIOFILE_ROOT
if(DEFINED ENV{AUDIOFILE_ROOT})
    list(APPEND _audiofile_search_paths $ENV{AUDIOFILE_ROOT})
endif()

# Priority 3: Standard system locations
list(APPEND _audiofile_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# Find the header file
find_path(AUDIOFILE_INCLUDE_DIR
    NAMES AudioFile.h
    PATHS ${_audiofile_search_paths}
    PATH_SUFFIXES include include/audiofile
    DOC "AudioFile include directory"
)

# Handle standard arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AudioFile
    FOUND_VAR AUDIOFILE_FOUND
    REQUIRED_VARS AUDIOFILE_INCLUDE_DIR
)

# Create imported target for modern CMake
if(AUDIOFILE_FOUND)
    if(NOT TARGET AudioFile::AudioFile)
        add_library(AudioFile::AudioFile INTERFACE IMPORTED)
        set_target_properties(AudioFile::AudioFile PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${AUDIOFILE_INCLUDE_DIR}"
        )
    endif()
endif()

# Clean up temporary variables
unset(_audiofile_search_paths)
unset(AUDIOFILE_INCLUDE_DIR)