@echo off
echo ========================================================
echo Starting Munchin' Backend API locally via .venv
echo ========================================================
echo.

:: Ensure pythonpath is set to locate back-end modules
set PYTHONPATH=%CD%\back-end

:: Activate the virtual environment
call .venv\Scripts\activate

:: Start the FastAPI server using uvicorn
python -m uvicorn api.main:app --host 127.0.0.1 --port 10800 --reload

pause
