@echo off

REM Set the base directories.
set "BASE_DIR=%~dp0"
set "BUILD_DIR=%BASE_DIR%_build"

set "COMMAND=%*"

REM Detect release or debug mode from the first argument, remove it from the command.
if /I "%~1"=="debug" (
    set "BUILD_TYPE=debug"
    set "COMMAND=%COMMAND:debug =%"
    if not exist "%BUILD_DIR%\debug\" (
        echo Debug build directory does not exist: %BUILD_DIR%\debug\
        echo Please run `build.bat all debug` first.
        exit /b 1
    )
) else if /I "%~1"=="release" (
    set "BUILD_TYPE=release"
    set "COMMAND=%COMMAND:release =%"
    if not exist "%BUILD_DIR%\release\" (
        echo Release build directory does not exist: %BUILD_DIR%\release\
        echo Please run `build.bat all release` first.
        exit /b 1
    )
)

REM If BUILD_TYPE is not set, check if only one of the build directories exists and set BUILD_TYPE accordingly.
if not defined BUILD_TYPE (
    if exist "%BUILD_DIR%\release\" (
        REM Default to release if at least release build exists.
        set "BUILD_TYPE=release"
    ) else if exist "%BUILD_DIR%\debug\" (
        REM Default to debug if only debug build exists.    
        set "BUILD_TYPE=debug"
    ) else (
        echo No build directory exists. Please run `build.bat all` first.
        exit /b 1
    )
)

set "PATH=%BUILD_DIR%\%BUILD_TYPE%\audio2x-sdk\bin;%PATH%"
set PYTHONPATH=%BASE_DIR%audio2x-common\scripts;%PYTHONPATH%

@REM Add CUDA bin dir to PATH if CUDA_PATH if defined
if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
) else (
    echo CUDA_PATH is not defined
    exit /b 1
)

@REM Add TensorRT lib dir to PATH if TENSORRT_ROOT_DIR if defined
if defined TENSORRT_ROOT_DIR (
     set "PATH=%TENSORRT_ROOT_DIR%\lib;%PATH%"
) else (
    echo TENSORRT_ROOT_DIR is not defined
    exit /b 1
)

REM If the first argument is not a file, try to find the corresponding executable in the build directory.
setlocal enabledelayedexpansion
for /f "tokens=1*" %%A in ("%COMMAND%") do set "FIRST_ARG=%%A"
if not exist %FIRST_ARG% (
    set "RELATIVE_PATH=%BUILD_DIR%\%BUILD_TYPE%"
    if exist !RELATIVE_PATH!\%FIRST_ARG% (
        call !RELATIVE_PATH!\%COMMAND%
    ) else (
        echo Error: Found neither an absolute path:
        echo   !FIRST_ARG!
        echo nor a relative path from the build directory:
        echo   !RELATIVE_PATH!\%FIRST_ARG%
        exit /b 1
    )
)

call %COMMAND%
