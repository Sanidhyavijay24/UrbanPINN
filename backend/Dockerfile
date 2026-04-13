# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (required for PyTorch and Trimesh/GeoPandas occasionally)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any packaging requirements without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend directory contents into the container
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI server via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
