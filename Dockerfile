# Base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed for C extensions (e.g., LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt-get/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . /app

# Expose Render default port
EXPOSE 10000

# Health check for Render / Docker
HEALTHCHECK CMD curl --fail http://localhost:10000/_stcore/health || exit 1

# Start the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]