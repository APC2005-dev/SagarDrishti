@echo off
rem Command Prompt entry point: "dev infra migrate" -> dev.ps1 (no execution-policy change needed)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0dev.ps1" %*
exit /b %ERRORLEVEL%
