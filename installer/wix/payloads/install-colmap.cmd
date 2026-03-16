@echo off
setlocal ENABLEEXTENSIONS

if "%~1"=="" (
  echo Usage: install-colmap.cmd ^<install-dir^> ^<colmap-zip^>
  exit /b 2
)

if "%~2"=="" (
  echo Usage: install-colmap.cmd ^<install-dir^> ^<colmap-zip^>
  exit /b 2
)

set "INSTALL_DIR=%~1"
set "ZIP_FILE=%~2"

if not exist "%ZIP_FILE%" (
  echo ERROR: COLMAP zip not found: "%ZIP_FILE%"
  exit /b 3
)

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%INSTALL_DIR%' -Force"
if errorlevel 1 (
  echo ERROR: Failed to extract COLMAP archive.
  exit /b 4
)

set "COLMAP_BAT=%INSTALL_DIR%\COLMAP.bat"
if not exist "%COLMAP_BAT%" (
  for /f "delims=" %%D in ('dir /b /ad "%INSTALL_DIR%"') do (
    if exist "%INSTALL_DIR%\%%D\COLMAP.bat" (
      xcopy "%INSTALL_DIR%\%%D\*" "%INSTALL_DIR%\" /E /I /Y >nul
    )
  )
)

if not exist "%COLMAP_BAT%" (
  echo ERROR: COLMAP.bat not found after extraction.
  exit /b 5
)

setx COLMAP_EXE "%COLMAP_BAT%" /M >nul 2>nul

echo COLMAP installed at "%INSTALL_DIR%"
echo COLMAP_EXE set to "%COLMAP_BAT%"
exit /b 0
