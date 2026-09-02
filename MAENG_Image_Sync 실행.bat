@echo off
chcp 65001 > nul
rem 사용자별 Python 설치 경로 자동 탐색 (%LOCALAPPDATA%\Programs\Python\Python3xx) — 없으면 py 런처
set "PYW="
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do if exist "%%~D\pythonw.exe" set "PYW=%%~D\pythonw.exe"
if not defined PYW set "PYW=pyw"
start "" "%PYW%" "%~dp0MAENG_Image_Sync.py" %*
