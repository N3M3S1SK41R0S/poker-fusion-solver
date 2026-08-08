@echo off
REM Poker Fusion Solver — lanceur Windows
REM Ordre : .venv du depot (fiable), puis uv, puis python du PATH
REM (attention : sur cette machine, le python du PATH est un venv tiers
REM sans numpy — d'ou la priorite au .venv).
cd /d "%~dp0python"
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -u -m pfs %*
) else (
    where uv >nul 2>nul
    if %errorlevel%==0 (
        uv run python -m pfs %*
    ) else (
        python -m pfs %*
    )
)
pause
