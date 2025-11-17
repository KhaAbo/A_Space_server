#!/bin/bash
# Quick script to test Docker setup

echo "🐳 Testing Docker Setup..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "   Please start Docker Desktop and try again."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check if model weights exist
echo "📦 Checking model weights..."
if [ -f "gaze-estimation-testing-main/gaze-estimation/weights/resnet50.pt" ]; then
    echo "✅ resnet50.pt found"
else
    echo "❌ resnet50.pt not found - run 'git lfs pull'"
    exit 1
fi

if [ -f "gaze-estimation-testing-main/gaze-estimation/weights/mobileone_s0_gaze.onnx" ]; then
    echo "✅ mobileone_s0_gaze.onnx found"
else
    echo "⚠️  mobileone_s0_gaze.onnx not found"
fi

echo ""
echo "🔨 Building Docker image..."
docker-compose build

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo ""
    echo "To start the API, run:"
    echo "  docker-compose up"
    echo ""
    echo "Or run in background:"
    echo "  docker-compose up -d"
else
    echo ""
    echo "❌ Build failed! Check errors above."
    exit 1
fi

