@echo off
echo ==========================================
echo Stopping Munchin' App & Cloudflare Tunnel
echo ==========================================

echo.
echo [1/2] Stopping website containers...
docker compose down

echo.
echo [2/2] Stopping user-space Cloudflare Tunnel...
taskkill /f /im cloudflared.exe >nul 2>&1

echo.
echo All services stopped.
echo ==========================================
pause
