# Use Python 3.14 full image (not slim) for better build tool support
FROM python:3.14.0

# Set working directory
WORKDIR /app

# Install additional system dependencies for image processing
# Full image already has build tools (gcc, g++, make, git, etc.)
RUN apt-get update && apt-get install -y \
    libtiff6 \
    libtiff-dev \
    libjpeg62-turbo \
    libjpeg62-turbo-dev \
    zlib1g-dev \
    libfreetype6-dev \
    libpng-dev \
    liblcms2-dev \
    libwebp-dev \
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
