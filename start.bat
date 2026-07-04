@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Starting Munchin' App and Cloudflare Tunnel
echo ===================================================

echo.
echo [1/3] Checking if Docker is running...
docker info >nul 2>&1
if %errorlevel% equ 0 goto docker_running

echo Docker is not running. Starting Docker Desktop...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

echo Waiting for Docker Desktop to be ready...
:wait_docker
ping 127.0.0.1 -n 6 >nul
docker info >nul 2>&1
if %errorlevel% equ 0 goto docker_started
echo Retrying connection to Docker...
goto wait_docker

:docker_started
echo.
echo Docker started successfully!
goto check_tunnel

:docker_running
echo Docker is already running.

:check_tunnel
echo.
echo [2/3] Checking Cloudflare Tunnel status...
sc query Cloudflared 2>nul | findstr /i "RUNNING" >nul
if %errorlevel% equ 0 goto tunnel_running

echo Cloudflared service is not running. Attempting to start service...
sc start Cloudflared >nul 2>&1
ping 127.0.0.1 -n 4 >nul
sc query Cloudflared 2>nul | findstr /i "RUNNING" >nul
if %errorlevel% equ 0 goto tunnel_started

echo Could not start Cloudflared Windows service (needs Admin privileges).
echo Starting user-space Cloudflare Tunnel in background...
start /b "" "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel run --token eyJhIjoiNDE3NzUzOTdiNGIxNWRjMmU2YTg0MjRjZDNjZWZkN2UiLCJ0IjoiMzgxNzMyMjktNGY5MC00NjFlLTk0YjctZDc1ZTQwYzBkODBjIiwicyI6IlpqQmxOV0poWVRJdE1tTTVOUzAwTkdFeUxUbG1ZalV0T1RGaE0yRXdOR1UxT0dFMyJ9 >nul 2>&1
echo Tunnel started in background.
goto start_containers

:tunnel_started
echo Cloudflared service started successfully.
goto start_containers

:tunnel_running
echo Cloudflare Tunnel is already active.

:start_containers
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
echo Startup complete.
