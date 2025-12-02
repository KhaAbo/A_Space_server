# Use Python 3.12 for better package compatibility
FROM python:3.12-slim

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements-docker.txt .

# Install Python dependencies
# Note: Using CPU version of PyTorch for Docker compatibility
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY api/ ./api/
COPY gaze-estimation-testing-main/ ./gaze-estimation-testing-main/

# Create directories for weights (will be mounted as volume)
RUN mkdir -p gaze-estimation-testing-main/gaze-estimation/weights

# Create storage directories
RUN mkdir -p storage/uploads storage/outputs

# Pre-download uniface (RetinaFace) model to avoid download on first run
# The model will be cached in /root/.uniface/models and can be persisted via volume mount
RUN python -c "import uniface; uniface.RetinaFace()" || echo "Uniface model pre-download (will use volume mount)"

# Copy startup script
COPY start_server.sh /app/start_server.sh
RUN chmod +x /app/start_server.sh

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run the API using startup script for better error visibility
CMD ["/app/start_server.sh"]

