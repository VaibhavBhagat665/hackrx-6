# Use a Python 3.9 slim base image for a smaller footprint
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file from the project root into the container
COPY requirements.txt .

# Install Python dependencies with a timeout to prevent potential issues
# Install torch separately to ensure a CPU-only version is used
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --timeout 600 -r requirements.txt

# Pre-download the embedding model to the image to avoid runtime downloads
# This fixes the "Out of memory" error by avoiding a large download at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy the application code from the project root
COPY app/ ./app/

# Expose the port the application will run on
EXPOSE 8000

# Use a proper Uvicorn command to bind to the host and port correctly
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
