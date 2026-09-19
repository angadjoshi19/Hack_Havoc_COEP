FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Pre-train/calibrate the model during the Docker build phase so it doesn't happen on startup
RUN python -c "from backend.model import load_or_initialize_model; load_or_initialize_model()"

# Expose port (default 8000, dynamically overridden by cloud hosts via $PORT)
EXPOSE 8000
ENV PORT=8000

CMD ["python", "start_servers.py"]

