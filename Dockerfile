# Use Python 3.14 slim image
FROM python:3.14.0-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by Pillow and matplotlib
RUN apt-get update && apt-get install -y \
    libtiff6 \
    libjpeg62-turbo \
    zlib1g \
    libfreetype6 \
    libpng16-16 \
    liblcms2-2 \
    libwebp7 \
    libopenjp2-7 \
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
