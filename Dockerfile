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
RUN pip install --no-cache-dir --timeout 600 -r requirements.txt

# Copy the application code from the project root
COPY app/ ./app/

# Expose the port the application will run on
EXPOSE 8000

# Set the command to run the application
CMD ["python", "-m", "app.main"]