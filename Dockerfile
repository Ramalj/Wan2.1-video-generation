FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

SHELL ["/bin/bash", "-c"]

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download model
COPY builder.py .
RUN python builder.py

# Copy handler
COPY handler.py .

# Start handler
CMD ["python", "-u", "handler.py"]
