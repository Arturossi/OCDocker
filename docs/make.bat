@ECHO OFF
SETLOCAL ENABLEEXTENSIONS

pushd %~dp0

REM --- Configuration ---
if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)
set SPHINXAUTOBUILD=sphinx-autobuild
set SOURCEDIR=source
set BUILDDIR=build
set IMAGE_SOURCE=OCDocker.png
set IMAGE_TARGET=%SOURCEDIR%\_static\OCDocker.png

REM --- Ensure sphinx-build is available ---
%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
    echo.
    echo The 'sphinx-build' command was not found. Make sure you have Sphinx
    echo installed, then set the SPHINXBUILD environment variable to point
    echo to the full path of the 'sphinx-build' executable. Alternatively, you
    echo may add the Sphinx directory to PATH.
    echo.
    echo If you don't have Sphinx installed, grab it from:
    echo https://www.sphinx-doc.org/
    exit /b 1
)

REM --- Copy asset if it exists ---
if exist "%IMAGE_SOURCE%" (
    if not exist "%SOURCEDIR%\_static" (
        mkdir "%SOURCEDIR%\_static"
    )
    copy /Y "%IMAGE_SOURCE%" "%IMAGE_TARGET%" >NUL
) else (
    echo Note: %IMAGE_SOURCE% not found, skipping image copy.
)

REM --- Run Sphinx ---
if "%1" == "" goto help

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%

:end
popd
ENDLOCAL
