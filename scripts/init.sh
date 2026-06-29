#!/bin/bash
set -e

echo "Waiting for MinIO..."
until curl -sf http://localhost:9000/minio/health/live >/dev/null 2>&1; do
    sleep 1
done

echo "Creating bucket: vn-food-images"
mc alias set local http://localhost:9000 minioadmin minioadmin 2>/dev/null || true
mc mb local/vn-food-images 2>/dev/null || true

echo "Init complete"