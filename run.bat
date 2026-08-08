@echo off
REM Poker Fusion Solver — lanceur Windows
cd /d "%~dp0python"
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run python -m pfs %*
) else (
    python -m pfs %*
)
pause
