# Use Python 3.14 slim image
FROM python:3.14.0-slim

# Set working directory
WORKDIR /app

# Install build dependencies and runtime libraries
# Build tools needed for compiling matplotlib, shap dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    pkg-config \
    git \
    libtiff6 \
    libtiff-dev \
    libjpeg62-turbo \
    libjpeg62-turbo-dev \
    zlib1g \
    zlib1g-dev \
    libfreetype6 \
    libfreetype6-dev \
    libpng16-16 \
    libpng-dev \
    liblcms2-2 \
    liblcms2-dev \
    libwebp7 \
    libwebp-dev \
    libopenjp2-7 \
    libopenjp2-7-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (Railway will override this with $PORT)
EXPOSE 8080

# Start command
CMD gunicorn app:app --bind 0.0.0.0:$PORT
