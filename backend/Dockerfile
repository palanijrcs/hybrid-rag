FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for compiling certain packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY backend/ ./backend/
COPY .env .env

# Expose FastAPI default port
EXPOSE 8000

# Start Uvicorn pointing to backend.app.main:app
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]