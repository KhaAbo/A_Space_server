# 🐳 Docker Quick Start - For Team Members

## Why Docker?

Docker ensures **everyone runs the exact same environment**, avoiding "works on my machine" issues. 

**Problem we're solving:** Python 3.13.5 has compatibility issues, but 3.13.2 works fine. With Docker, everyone gets the working version automatically!

## Prerequisites

1. **Install Docker Desktop**: https://www.docker.com/products/docker-desktop
2. **Install Git LFS**: `git lfs install`

## 3-Step Setup

```bash
# 1. Clone the repo (if you haven't already)
git clone https://github.com/KhaAbo/A_Space_server
cd A_Space_server

# 2. Pull model weights with Git LFS
git lfs pull

# 3. Set up Discord webhook (optional but recommended)
# Copy the example file and add your Discord webhook URL
cp api/env.example api/.env
# Edit api/.env and add your Discord webhook URL

# 4. Start the API
docker-compose up --build
```

That's it! The API is now running at http://localhost:8000 🎉

## Common Commands

```bash
# Start API (builds automatically)
docker-compose up --build

# Run in background
docker-compose up -d

# Stop API
docker-compose down

# View logs
docker-compose logs -f

# Check if it's working
curl http://localhost:8000/api/health
```

## Testing the API

1. Open browser: http://localhost:8000/docs
2. Try the interactive API documentation
3. Upload a test video and see it process!

## Troubleshooting

### "Docker is not running"
→ Start Docker Desktop application

### "Model weights not found"
→ Run: `git lfs pull`

### "Port 8000 already in use"
→ Stop other services or change port in `docker-compose.yml`

## Need More Info?

- Full Docker guide: [DOCKER_SETUP.md](DOCKER_SETUP.md)
- API documentation: [API_README.md](API_README.md)
- General info: [README.md](README.md)

---

**Note:** If you prefer local setup without Docker, see [README.md](README.md) but make sure you use Python 3.10-3.13.2 (NOT 3.13.5+).

