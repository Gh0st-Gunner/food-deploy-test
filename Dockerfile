FROM python:3.11-slim-bookworm AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Set PYTHONPATH to include the back-end folder so python can find modules
ENV PYTHONPATH=/app/back-end

# Expose API port
EXPOSE 10800

# Default: run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10800", "--workers", "4"]