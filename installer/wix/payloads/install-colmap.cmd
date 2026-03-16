@echo off
setlocal ENABLEEXTENSIONS

if "%~1"=="" (
  echo Usage: install-colmap.cmd ^<colmap-zip^> OR install-colmap.cmd ^<install-dir^> ^<colmap-zip^>
  exit /b 2
)

set "DEFAULT_INSTALL_DIR=%ProgramW6432%\Bimba3D\third_party\colmap"
if "%DEFAULT_INSTALL_DIR%"=="\Bimba3D\third_party\colmap" set "DEFAULT_INSTALL_DIR=%ProgramFiles%\Bimba3D\third_party\colmap"

if "%~2"=="" (
  set "INSTALL_DIR=%DEFAULT_INSTALL_DIR%"
  set "ZIP_FILE=%~1"
) else (
  set "INSTALL_DIR=%~1"
  set "ZIP_FILE=%~2"
)

if "%INSTALL_DIR:~0,1%"=="[" set "INSTALL_DIR=%DEFAULT_INSTALL_DIR%"

if not exist "%ZIP_FILE%" (
  echo ERROR: COLMAP zip not found: "%ZIP_FILE%"
  exit /b 3
)

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath '%ZIP_FILE%' -DestinationPath '%INSTALL_DIR%' -Force"
if errorlevel 1 (
  echo ERROR: Failed to extract COLMAP archive.
  exit /b 4
)

set "COLMAP_BAT=%INSTALL_DIR%\COLMAP.bat"
if not exist "%COLMAP_BAT%" (
  for /f "usebackq delims=" %%F in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-ChildItem -LiteralPath '%INSTALL_DIR%' -Filter 'COLMAP.bat' -Recurse -File -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty FullName; if ($p) { Write-Output $p }"`) do (
    set "COLMAP_BAT=%%F"
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
