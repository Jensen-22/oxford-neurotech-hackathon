@echo off
REM UnSway launcher for Windows using UnicornPy (direct headset access, no Unicorn LSL needed).
REM Copy of START_UNSWAY_WINDOWS.bat with --source unicornpy.
set "UNICORNPY_LIB=%USERPROFILE%\OneDrive\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib"
if not exist "%UNICORNPY_LIB%\UnicornPy.pyd" set "UNICORNPY_LIB=%USERPROFILE%\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn Python\Lib"
set "PYTHONPATH=%UNICORNPY_LIB%;%PYTHONPATH%"
set "PATH=%UNICORNPY_LIB%;%PATH%"
REM UnSway launcher for Windows. Double-click this file.
REM Needs: Python 3.10/3.11 (python.org, tick "Add to PATH"), Unicorn Suite with the "Unicorn LSL" app.
cd /d "%~dp0"
setlocal enabledelayedexpansion

if not exist .venv (
  echo First-time setup ^(3-5 min^)...
  python -m venv .venv || (echo Python not found. Install Python 3.11 from python.org and tick "Add to PATH". & pause & exit /b)
  call .venv\Scripts\activate.bat
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt pylsl websockets pyserial pandas scikit-learn pillow matplotlib scipy
  python -m pip install -q psychopy || echo PsychoPy failed to install - send Claude the error.
) else (
  call .venv\Scripts\activate.bat
)

:menu
echo.
echo ======= UnSway (Windows, UnicornPy) =======
echo  FIRST: turn Bluetooth and the headset on (no Unicorn LSL needed).
echo  1) Headset check (1 min)
echo  2) Live dashboards only
echo  3) ODDBALL pilot (4 min)
echo  4) PARTICIPANT session (~12 min)
echo  5) Analyse ALL participants
echo  q) Quit
set /p c=Choose:
if "%c%"=="1" python stream_unicorn.py --source unicornpy --check & goto menu
if "%c%"=="2" (
  start "UnSway stream" cmd /k python stream_unicorn.py --source unicornpy --name live.csv
  timeout /t 6 >nul & call :dash & goto menu
)
if "%c%"=="3" (
  set /p id=Oddball ID [OB01]:
  if "!id!"=="" set id=OB01
  start "UnSway stream" cmd /k python stream_unicorn.py --source unicornpy --name !id!.csv
  timeout /t 6 >nul & call :dash
  python mindcart\experiment.py --pid !id! --oddball
  echo Stopping the stream window... close it if it is still open.
  taskkill /fi "WINDOWTITLE eq UnSway stream*" >nul 2>&1
  python -m mindcart.erp recordings\!id!.csv --oddball
  goto menu
)
if "%c%"=="4" (
  set /p id=Participant ID ^(e.g. P01^):
  start "UnSway stream" cmd /k python stream_unicorn.py --source unicornpy --name !id!.csv
  timeout /t 6 >nul & call :dash
  python mindcart\experiment.py --pid !id! --items stimuli\items.csv --img-dir stimuli\norm
  taskkill /fi "WINDOWTITLE eq UnSway stream*" >nul 2>&1
  python -m mindcart.erp recordings\!id!.csv
  goto menu
)
if "%c%"=="5" (
  set files=
  for %%f in (recordings\P*.csv) do set files=!files! %%f
  python -m mindcart.features !files!
  python -m mindcart.erp !files!
  python -m mindcart.models !files! --items stimuli\items.csv
  start "" recordings\results.png
  goto menu
)
if /i "%c%"=="q" exit /b
goto menu

:dash
set "u=%CD:\=/%"
start "" chrome "file:///!u!/dashboard/pipeline.html?ws=ws://localhost:8766" || start "" msedge "file:///!u!/dashboard/pipeline.html?ws=ws://localhost:8766"
start "" chrome "file:///!u!/dashboard/brainwaves.html?ws=ws://localhost:8766"
start "" chrome "file:///!u!/dashboard/mindcart.html?ws=ws://localhost:8765" || start "" msedge "file:///!u!/dashboard/mindcart.html?ws=ws://localhost:8765"
exit /b
