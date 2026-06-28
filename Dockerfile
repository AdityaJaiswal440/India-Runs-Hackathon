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

# Run the pipeline to generate submission.csv as default entrypoint
CMD ["python", "run_pipeline.py"]