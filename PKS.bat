@echo off
REM ─────────────────────────────────────────────────────────────────────
REM  PKS — Poker Solveur : lanceur du raccourci bureau.
REM  Demarre le serveur local (127.0.0.1 uniquement) et ouvre le navigateur.
REM  Fermer cette fenetre, ou Ctrl+C, arrete le logiciel.
REM ─────────────────────────────────────────────────────────────────────
title PKS - Poker Solveur
cd /d "%~dp0python"

if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" -u -m pfs %*
    goto :fin
)

where uv >nul 2>nul
if %errorlevel%==0 (
    uv run python -m pfs %*
    goto :fin
)

python -u -m pfs %*

:fin
echo.
echo PKS est arrete. Appuie sur une touche pour fermer.
pause >nul
