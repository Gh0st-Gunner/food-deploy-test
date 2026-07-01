@echo off
echo ========================================================
echo Starting Munchin' Celery Worker locally via .venv
echo ========================================================
echo.

:: Ensure pythonpath is set to locate back-end modules
set PYTHONPATH=%CD%\back-end

:: Activate the virtual environment
call .venv\Scripts\activate

:: Start Celery worker
celery -A workers.celery_app worker --loglevel=info

pause
