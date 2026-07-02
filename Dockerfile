# Use official lightweight Python 3.11 base image
FROM python:3.11-slim

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/app/src

# Working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install build tools, dependencies, then remove build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.12.1+cpu && \
    pip install --no-cache-dir \
        --upgrade-strategy only-if-needed \
        -r requirements.txt && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the application
COPY . .
# Copy precomputed embeddings into the container for reproducibility
RUN mkdir -p data/embeddings
COPY artifacts/embeddings.fp16.npz data/embeddings/embeddings.fp16.npz
COPY artifacts/candidate_ids.json data/embeddings/candidate_ids.json

# Copy the committed submission artifact so /reproduce endpoint can serve it
COPY era.csv era.csv

# Expose port for the Sandbox API
EXPOSE 8000

# Start the API
CMD ["uvicorn", "sandbox.app:app", "--host", "0.0.0.0", "--port", "8000"]