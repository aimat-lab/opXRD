@echo off
setlocal enabledelayedexpansion

rem Define the root directory where the script and build should be located
set "ROOT_DIR=%USERPROFILE%\xrd_data_collect"

if not exist "%ROOT_DIR%" (
    mkdir "%ROOT_DIR%"
)

set "DIST_DIR=%ROOT_DIR%\dist"
set "BUILD_DIR=%ROOT_DIR%\build"
@REM for /f %%i in ('cd') do set "script_dir=%%~fi"

pyinstaller "prod_run.spec" --distpath "%DIST_DIR%" --workpath "%BUILD_DIR%"
echo -> Built executable under %ROOT_DIR%\dist
echo done