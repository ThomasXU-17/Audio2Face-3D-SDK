@echo off

@REM Reset the errorlevel. If invoked by CMake, it may be set to a non-zero value.
@REM This can cause issues with packman commands that rely on errorlevel.
set errorlevel=0

set BASE_DIR=%~dp0

set BUILD_CONFIG=release
@REM Check if a build configuration was provided as an argument
if not "%1"=="" (
    set BUILD_CONFIG=%1
)
if "%PYTHON%"=="" SET "PYTHON=tools\packman\python.bat"
if "%PACKMAN%"=="" SET "PACKMAN=%BASE_DIR%tools\packman\packman"

@REM Pull dependencies using packman
call %PACKMAN% pull -t config=%BUILD_CONFIG% --platform windows-x86_64 %BASE_DIR%deps\build-deps.packman.xml
if errorlevel 1 (
    echo Failed to pull dependencies in build-deps.packman.xml
    exit /b 1
)
call %PACKMAN% pull -t config=%BUILD_CONFIG% --platform windows-x86_64 %BASE_DIR%deps\target-deps.packman.xml
if errorlevel 1 (
    echo Failed to pull dependencies in target-deps.packman.xml
    exit /b 1
)
