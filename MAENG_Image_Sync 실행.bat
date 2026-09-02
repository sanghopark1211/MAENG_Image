@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

rem ------------------------------------------------------------------
rem  Python 자동 탐색.
rem  주의: 단순히 마지막 폴더를 고르면 안 된다 — Python310/Python38 처럼
rem  여러 버전이 깔려 있으면 패키지가 없는 쪽이 잡혀 pythonw 가 창도 없이
rem  조용히 죽는다. 그래서 '실제로 import 되는' 것을 골라야 한다.
rem ------------------------------------------------------------------
set "PYW="
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if not defined PYW if exist "%%~D\pythonw.exe" (
        "%%~D\python.exe" -c "import numpy,cv2,PIL,pycine" >nul 2>&1
        if !errorlevel! equ 0 set "PYW=%%~D\pythonw.exe"
    )
)

rem py 런처도 후보로
if not defined PYW (
    py -c "import numpy,cv2,PIL,pycine" >nul 2>&1
    if !errorlevel! equ 0 set "PYW=pyw"
)

if not defined PYW (
    echo.
    echo   [MAENG_Image_Sync] 실행할 수 있는 Python 을 찾지 못했습니다.
    echo   필요한 패키지: numpy, opencv-python, pillow, pycine
    echo.
    echo   설치 명령 ^(명령 프롬프트에서^):
    echo       pip install numpy opencv-python pillow pycine
    echo.
    pause
    exit /b 1
)

start "" "%PYW%" "%~dp0MAENG_Image_Sync.py" %*
