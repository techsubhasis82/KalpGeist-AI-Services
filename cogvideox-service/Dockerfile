# Use PyTorch 2.7.0 with CUDA 12.8 for RTX 5090 support
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set working directory
WORKDIR /app

# Install system dependencies including OpenGL libraries
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-extra \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Expose port 5000
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_USE_CUDA_DSA=1

# Pin CUDA architecture for RTX 5090 compatibility (sm_120)
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
ENV CUDA_VISIBLE_DEVICES=0

# Run the application
CMD ["python", "app.py"]