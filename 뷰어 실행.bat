@echo off
chcp 65001 > nul
set "PY="
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%~D\python.exe" set "PY=%%~D\python.exe"
if not defined PY set "PY=py"
"%PY%" "%~dp0image_compare_app.py" %*
pause
