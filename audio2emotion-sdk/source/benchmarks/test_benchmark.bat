@echo off

set "COMMAND=%*"

@REM At least one arg should be passed
if "%1"=="" (
    echo "Usage: test_benchmark.bat <executable> [args..]"
    exit /b 1
)

REM Define the list of arguments
set args[0]=--benchmark_filter=BM_InteractiveExecutorBatch/BatchSize:0/InferencesToSkip:0/Classifier:1/real_time
set args[1]=--benchmark_filter=BM_InteractiveExecutorLayer/Layer:0/Classifier:1/real_time
set args[2]=--benchmark_filter=BM_ExecutorPartial/ActiveTracks:1/iterations:10/real_time

setlocal enabledelayedexpansion

REM Iterate over the arguments and execute the benchmark command
for /L %%i in (0,1,2) do (
    set "current_arg=!args[%%i]!"
    echo "%%i"
    echo Running: %COMMAND% !current_arg!
    call %COMMAND% !current_arg!

    if !errorlevel! neq 0 (
        exit /b !errorlevel!
    )
)
