FROM python:3.11-slim

WORKDIR /app

# Copy dependency definitions and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy Streamlit frontend code
COPY frontend/ ./frontend/
COPY .env .env

# Expose Streamlit default port
EXPOSE 8501

# Streamlit healthcheck configuration
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Run Streamlit bound to 0.0.0.0
ENTRYPOINT ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]