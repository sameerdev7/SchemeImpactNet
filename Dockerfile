FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/db models reports/figures \
    && chown -R appuser:appuser /app

USER appuser

# HF Spaces exposes port 7860
EXPOSE 7860

# Entrypoint: generate synthetic data, seed DB, start both services
CMD ["bash", "hf_start.sh"]
