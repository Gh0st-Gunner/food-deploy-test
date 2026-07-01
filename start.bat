@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Starting Munchin' App & Cloudflare Tunnel
echo ==========================================

echo.
echo [1/3] Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not running. Starting Docker Desktop...
    docker desktop start >nul 2>&1
    
    echo Waiting for Docker Desktop to be ready...
    :wait_docker
    timeout /t 2 >nul
    docker info >nul 2>&1
    if %errorlevel% neq 0 (
        echo. | set /p ="."
        goto wait_docker
    )
    echo.
    echo Docker started successfully!
) else (
    echo Docker is already running.
)

echo.
echo [2/3] Checking Cloudflare Tunnel status...
sc query Cloudflared 2>nul | findstr /i "RUNNING" >nul
if %errorlevel% neq 0 (
    echo Cloudflared service is not running. Attempting to start service...
    sc start Cloudflared >nul 2>&1
    timeout /t 3 >nul
    sc query Cloudflared 2>nul | findstr /i "RUNNING" >nul
    if %errorlevel% neq 0 (
        echo Could not start Cloudflared Windows service (needs Admin privileges).
        echo Starting user-space Cloudflare Tunnel in background...
        start /b "" "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel run --token eyJhIjoiNDE3NzUzOTdiNGIxNWRjMmU2YTg0MjRjZDNjZWZkN2UiLCJ0IjoiMzgxNzMyMjktNGY5MC00NjFlLTk0YjctZDc1ZTQwYzBkODBjIiwicyI6IlpqQmxOV0poWVRJdE1tTTVOUzAwTkdFeUxUbG1ZalV0T1RGaE0yRXdOR1UxT0dFMyJ9 >nul 2>&1
        echo Tunnel started in background.
    ) else (
        echo Cloudflared service started successfully.
    )
) else (
    echo Cloudflare Tunnel is already active.
)

echo.
echo [3/3] Starting website containers...
docker compose up -d

echo.
echo ===================================================
echo Munchin' App is ready!
echo.
echo [Munchin' User Interface]
echo Local URL:    http://localhost:10800
echo External URL: https://munchin.thegunner.uk
echo.
echo [Admin Dashboard]
echo Local URL:    http://localhost:10800/admin.html
echo External URL: https://munchin.thegunner.uk/admin.html
echo ===================================================
echo.
pause
