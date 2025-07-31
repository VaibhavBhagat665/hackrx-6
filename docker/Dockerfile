FROM python:3.9-slim

# set working directory
WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# copy requirements
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application
COPY app/ ./app/

# expose port
EXPOSE 8000

# run application
CMD ["python", "-m", "app.main"]
