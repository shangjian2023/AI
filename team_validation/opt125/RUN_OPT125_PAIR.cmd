@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_opt125_pair.ps1" %*
set EXIT_CODE=%ERRORLEVEL%
echo.
if not "%EXIT_CODE%"=="0" (
  echo Validation stopped with exit code %EXIT_CODE%.
  echo Keep the folder unchanged and send the logs directory to the captain.
) else (
  echo Validation completed. Send RETURN_TO_CAPTAIN_*.zip to the captain.
)
pause
exit /b %EXIT_CODE%
