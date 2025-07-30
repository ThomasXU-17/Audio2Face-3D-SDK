@echo off

@REM Set the default build configuration to release
set BUILD_CONFIG=release
@REM Set the default build project to all
set BUILD_PROJECT=all

@REM Select the build configuration and the project to build in arbitrary order.
if "%1"=="release" (
    set BUILD_CONFIG=%1
    if not "%2"=="" (
        set BUILD_PROJECT=%2
    )
) else if "%1"=="debug" (
    set BUILD_CONFIG=%1
    if not "%2"=="" (
        set BUILD_PROJECT=%2
    )
) else (
    if not "%1"=="" (
        set BUILD_PROJECT=%1
    )
    if not "%2"=="" (
        set BUILD_CONFIG=%2
    )
)

set ROOT_DIR=%~dp0
set PATH=%PATH%;%ROOT_DIR%_deps\build-deps\ninja

set CMAKE=%ROOT_DIR%_deps\build-deps\cmake\bin\cmake.exe

@REM Set the default build configuration to release
set BUILD_CONFIG=release

@REM Check if a build configuration was provided as an argument
if not "%2"=="" (
    set BUILD_CONFIG=%2
)

set BUILD_DIR=%ROOT_DIR%_build\%BUILD_CONFIG%

set BUILD_PROJECT="all"
if "%1"=="clean" (
    if exist "%BUILD_DIR%" (
        rmdir /s /q "%BUILD_DIR%"
    )
    echo Removed %BUILD_DIR%
    exit /b 0
) else if not "%1"=="" (
    set BUILD_PROJECT=%1
)

if not exist %ROOT_DIR%_deps (
    echo Dependencies not found. Please run .\fetch_deps.bat %BUILD_CONFIG% first.
    exit /b 1
)

if not exist %BUILD_DIR% (
    mkdir %BUILD_DIR%
)

@REM Find the latest Visual Studio installation path using vswhere
for /f "delims=" %%i in ('"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -requires Microsoft.Component.MSBuild -property installationPath') do set "VS_PATH=%%i"

@REM If vswhere fails to detect Visual Studio, set the path manually below:
@REM Example for Visual Studio 2022 Community Edition:
@REM set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community

@REM Check if VS_PATH was set successfully
if not defined VS_PATH (
    echo Visual Studio installation not found.
    exit /b 1
)

@REM Call vcvarsall.bat with x64 argument
call "%VS_PATH%\VC\Auxiliary\Build\vcvarsall.bat" x64 > %BUILD_DIR%\vsdevcmd.trace.txt 2>&1

%CMAKE% -B %BUILD_DIR% -G Ninja -S . -DCMAKE_BUILD_TYPE=%BUILD_CONFIG%

%CMAKE% --build %BUILD_DIR% --target %BUILD_PROJECT% --config %BUILD_CONFIG% --parallel
