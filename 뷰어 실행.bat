@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

rem 패키지가 실제로 설치된 Python 을 고른다 (버전이 여러 개일 때 오작동 방지)
set "PY="
for /d %%D in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if not defined PY if exist "%%~D\python.exe" (
        "%%~D\python.exe" -c "import numpy,cv2,PIL,pycine" >nul 2>&1
        if !errorlevel! equ 0 set "PY=%%~D\python.exe"
    )
)
if not defined PY (
    py -c "import numpy,cv2,PIL,pycine" >nul 2>&1
    if !errorlevel! equ 0 set "PY=py"
)
if not defined PY (
    echo.
    echo   필요한 패키지를 갖춘 Python 을 찾지 못했습니다.
    echo       pip install numpy opencv-python pillow pycine
    echo.
    pause
    exit /b 1
)

"%PY%" "%~dp0image_compare_app.py" %*
pause
