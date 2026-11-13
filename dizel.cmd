@echo off
rem Dizel launcher - asks which UI unless you are already inside one.
chcp 65001 >nul
setlocal
set START=%CD%
pushd "%~dp0"

if /i "%START%"=="%CD%\inference\gui" goto gui
if /i "%START%"=="%CD%\inference\tui" goto tui
if /i "%START%"=="%CD%\inference" goto tui
echo.
rem Several Python installs may be on PATH; only one has the dependencies.
rem Resolved before the menu: the chooser itself needs Textual.
set PYEXE=
py -3 -c "import textual" >nul 2>nul
if not errorlevel 1 set PYEXE=py -3
if not defined PYEXE (
    python -c "import textual" >nul 2>nul
    if not errorlevel 1 set PYEXE=python
)

if not defined PYEXE goto plain_choice

rem A staged TUI update is copied in before any app module is imported. The
rem previous session wrote it to .dizel\updates; a failure leaves the old app.
%PYEXE% -m dizel_core.updater apply --quiet

rem A real chooser, built with the same Textual stack as the terminal client.
rem Exit codes: 1 terminal, 2 desktop, 3 cancelled, 4 resume last chat.
%PYEXE% -m inference.tui.launcher
if errorlevel 4 goto resume
if errorlevel 3 goto done
if errorlevel 2 goto gui
goto tui

:plain_choice
rem Fallback when Textual is missing: the original one-line menu.
echo.
echo   [38;5;75m◆[0m  [97mDizel[0m
echo   [90m──────────────────────────────────────────────────────────────[0m
echo.
echo    [97m1[0m  Terminal UI   [90mkeyboard-first, lives in the shell[0m
echo    [97m2[0m  Desktop app   [90mfull window, animations[0m
echo.
echo   [90m──────────────────────────────────────────────────────────────[0m
echo.
choice /C 12 /N /M "   [90mchoose[0m [1/2] "
if errorlevel 2 goto gui
goto tui

:tui
if not defined PYEXE goto no_python
%PYEXE% -m inference.tui.main %*
goto done

:resume
if not defined PYEXE goto no_python
%PYEXE% -m inference.tui.main --resume %*
goto done

:gui
if not exist "%~dp0inference\gui" (
echo.
echo   !  This package includes the terminal client only.
echo   ──────────────────────────────────────────────────────────────
echo   desktop  download the Dizel installer from the releases page
echo.
pause
goto done
)
set "LOG=%~dp0logs\gui.log"
if not exist "%~dp0logs" mkdir "%~dp0logs"
cd /d "%~dp0inference\gui"
title Dizel - starting desktop
set "VER="
for /f "delims=" %%v in ('node -p "require('./package.json').version" 2^>nul') do set "VER= %%v"
echo.
echo   [38;5;75m◆[0m  [97mDizel[0m[90m%VER%  ·  desktop[0m
echo   [90m──────────────────────────────────────────────────────────────[0m
echo.
echo   [90mwindow   [0mopening in a few seconds
echo   [90mlogs     [0mlogs\gui.log
echo   [90mstop     [0mclose the window, or press Ctrl+C here
echo.
if not exist node_modules (
echo   [90minstall  [0mGUI dependencies, first run only
    call npm install > "%LOG%" 2>&1
echo   [38;5;78m✓[0m  dependencies ready
echo.
)
where /q powershell
if errorlevel 1 goto gui_plain_start
if not exist "%~dp0scripts\gui_loader.ps1" goto gui_plain_start
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\gui_loader.ps1" -LogPath "%LOG%" -WorkDir "%CD%"
goto gui_started

:gui_plain_start
echo   [90mstatus   [0mstarting vite + electron
echo.
call npm run dev > "%LOG%" 2>&1

:gui_started
title Dizel
rem concurrently -k kills vite the moment electron exits, so a normal close
rem still returns non-zero. Electron own exit code is the honest signal.
findstr /C:"electron . exited with code 0" "%LOG%" >nul
if errorlevel 1 (
echo.
echo   [91m![0m  GUI exited with an error
echo   [90m──────────────────────────────────────────────────────────────[0m
echo   [90mlog      [0mlogs\gui.log
echo   [90mnext     [0mopen the log, or rerun from a terminal for the traceback
echo.
    pause
) else (
echo   [38;5;78m✓[0m  window closed
)
goto done

:no_python
echo.
echo   [91m![0m  No Python with the required packages was found.
echo   [90m──────────────────────────────────────────────────────────────[0m
echo   [90mfix      [0mpip install -r requirements.txt
echo.
pause
goto done

:done
popd
exit /b %errorlevel%
