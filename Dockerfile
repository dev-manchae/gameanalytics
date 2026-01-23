FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub to download models
RUN pip install huggingface_hub

# Download the model using Python directly to avoid CLI PATH issues
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='manchae86/steam-review-roberta', local_dir='/app/model', local_dir_use_symlinks=False)"

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Render uses PORT env variable)
EXPOSE 10000

# Run with gunicorn - use PORT env variable for Render, fallback to 10000
CMD gunicorn -b 0.0.0.0:${PORT:-10000} --timeout 120 app:server