# Docker Setup Guide

This guide will help you run the Gaze Estimation API using Docker, ensuring a consistent environment across all machines.

## Prerequisites

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
   - Download: https://www.docker.com/products/docker-desktop

2. **Git LFS** for model weights
   - Install: https://git-lfs.com/

## First-Time Setup

```bash
# 1. Clone the repository
git clone https://github.com/KhaAbo/A_Space_server
cd A_Space_server

# 2. Install Git LFS and pull model weights
git lfs install
git lfs pull

# 3. Verify model weights are downloaded
ls -lh gaze-estimation-testing-main/gaze-estimation/weights/
# Should show: resnet50.pt, mobileone_s0_gaze.onnx, resnet18_gaze.onnx
```

## Running the API

### Option 1: Docker Compose (Recommended)

```bash
# Start the API (builds image automatically)
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the API
docker-compose down
```

### Option 2: Manual Docker Commands

```bash
# Build the image
docker build -t gaze-api .

# Run the container
docker run -d \
  --name gaze-api \
  -p 8000:8000 \
  -v $(pwd)/storage:/app/storage \
  gaze-api

# View logs
docker logs -f gaze-api

# Stop and remove container
docker stop gaze-api
docker rm gaze-api
```

## Testing the API

Once running, test the API:

```bash
# Check health
curl http://localhost:8000/api/health

# Or visit in browser
# http://localhost:8000/docs
```

## Common Issues & Solutions

### Issue: Model weights not found

**Solution:**
```bash
# Make sure Git LFS pulled the weights
git lfs pull

# Verify files are not empty pointers
file gaze-estimation-testing-main/gaze-estimation/weights/resnet50.pt
# Should show "data" or "PyTorch model", not "ASCII text"
```

### Issue: Port 8000 already in use

**Solution:**
```bash
# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Use 8001 instead

# Or stop the conflicting service
docker ps  # Find what's using port 8000
```

### Issue: Out of disk space

**Solution:**
```bash
# Clean up old Docker images
docker system prune -a

# Remove unused volumes
docker volume prune
```

## Development Workflow

### Making code changes

1. Edit files in `api/` directory
2. Rebuild and restart:
```bash
docker-compose up --build
```

### Hot reload (development mode)

The `docker-compose.yml` already mounts the `api/` directory, so changes will be reflected automatically with uvicorn's `--reload` flag.

To enable:
```yaml
# In docker-compose.yml, change CMD to:
command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Production Deployment

For production, consider:

1. **Remove volume mounts** for source code
2. **Use environment variables** for configuration
3. **Set restart policy:**
```yaml
restart: always
```
4. **Add resource limits:**
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

## GPU Support (Optional)

If you have NVIDIA GPU and want to use it:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Create `Dockerfile.gpu`:
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
# ... rest of Dockerfile
```

3. Update `docker-compose.yml`:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Troubleshooting

### Container keeps restarting

```bash
# Check logs for errors
docker-compose logs -f

# Common causes:
# - Missing model weights
# - Port already in use
# - Insufficient memory
```

### Cannot connect to API

```bash
# Verify container is running
docker ps

# Check if port is mapped correctly
docker port gaze-api

# Test from inside container
docker exec -it gaze-api curl http://localhost:8000/api/health
```

## Useful Commands

```bash
# Enter container shell
docker exec -it gaze-api bash

# Check container resource usage
docker stats gaze-api

# Inspect container
docker inspect gaze-api

# Copy files from container
docker cp gaze-api:/app/storage/outputs ./outputs

# View container environment variables
docker exec gaze-api env
```

## Need Help?

- Check main [README.md](README.md) for general documentation
- Check [API_README.md](API_README.md) for API reference
- View API docs at http://localhost:8000/docs when running

