# Azure ML custom environment container for Video Editorial Intelligence
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies including FFmpeg for audio/video extraction
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY video-editorial-env.yml /tmp/video-editorial-env.yml
RUN conda env update --name base --file /tmp/video-editorial-env.yml && conda clean -afy

WORKDIR /workspace

