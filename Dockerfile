# Use official lightweight Python 3.11 base image
FROM python:3.11-slim

# Set environment variables for clean execution
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# Set working directory inside container
WORKDIR /app

# Install system dependencies if required (e.g., git, build essentials)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase into the container
COPY . .

# Setup embeddings to satisfy the canonical path guard
RUN mkdir -p data/embeddings && \
    cp artifacts/embeddings.fp16.npz data/embeddings/ && \
    cp artifacts/candidate_ids.json data/embeddings/

# Start the Interactive Sandbox API as default entrypoint
CMD ["uvicorn", "sandbox.app:app", "--host", "0.0.0.0", "--port", "8000"]