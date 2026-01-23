FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub specifically to download models efficiently
RUN pip install huggingface_hub

# Download the model files to /app/model during the build.
# This ensures the model is baked into the image and doesn't need to download at runtime.
RUN huggingface-cli download manchae86/steam-review-roberta --local-dir /app/model --local-dir-use-symlinks False

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Render uses PORT env variable)
EXPOSE 10000

# Run with gunicorn
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --timeout 120 app:server