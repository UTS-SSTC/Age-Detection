# Use CUDA base image
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA to install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv && \
    rm -rf /var/lib/apt/lists/*

# Set up a virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip in the virtual environment
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements files
COPY requirements/requirements-normal.txt .
COPY requirements/requirements-pytorch.txt .

# Install PyTorch dependencies
RUN pip install --no-cache-dir -r requirements-pytorch.txt

# Install normal dependencies
RUN pip install --no-cache-dir -r requirements-normal.txt

# Install Flask (for demo application)
RUN pip install --no-cache-dir flask

# Copy code to container
COPY . .

# Create necessary directories
RUN mkdir -p data/processed data/augmented data/verified_image data/wiki models

# Expose Flask service port
EXPOSE 5000

# Set environment variables
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONPATH=/app

# Startup command
CMD ["python", "demo/demo.py"]