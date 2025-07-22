# SadTalker Service Dockerfile - OFFICIAL INTEGRATION
# Production-ready with official SadTalker for maximum accuracy
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install SYSTEM dependencies for official SadTalker
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libsndfile1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Clone OFFICIAL SadTalker repository
RUN git clone https://github.com/OpenTalker/SadTalker.git /app/SadTalker

# Fix NumPy compatibility issues (np.float deprecated + array creation)
RUN sed -i 's/np.float/np.float64/g' /app/SadTalker/src/face3d/util/my_awing_arch.py
RUN sed -i 's/trans_params = np.array(\[w0, h0, s, t\[0\], t\[1\]\])/trans_params = np.array([float(w0), float(h0), float(s), float(t[0]), float(t[1])])/g' /app/SadTalker/src/face3d/util/preprocess.py

# Install PyTorch first (compatible versions)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install official SadTalker requirements
WORKDIR /app/SadTalker
RUN pip install --no-cache-dir -r requirements.txt

# Copy your custom wrapper files
WORKDIR /app
COPY requirements.txt .
COPY . .

# Install your custom requirements (RunPod integration)
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for output and models
RUN mkdir -p /app/output /app/models /app/SadTalker/checkpoints

# Set environment variables
ENV PYTHONPATH=/app:/app/SadTalker
ENV PYTHONUNBUFFERED=1

# Download official SadTalker models on build (optional)
# RUN cd /app/SadTalker && bash scripts/download_models.sh

# Run the application
CMD ["python", "app.py"]