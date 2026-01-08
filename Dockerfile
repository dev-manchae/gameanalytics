FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Render uses PORT env variable, HuggingFace uses 7860)
EXPOSE 10000

# Run with gunicorn - use PORT env variable for Render, fallback to 10000
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --timeout 120 app:server

