@echo off
REM Quick script to test Docker setup

echo 🐳 Testing Docker Setup...
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running!
    echo    Please start Docker Desktop and try again.
    exit /b 1
)

echo ✅ Docker is running
echo.

REM Check if model weights exist
echo 📦 Checking model weights...
if exist "gaze-estimation-testing-main\gaze-estimation\weights\resnet50.pt" (
    echo ✅ resnet50.pt found
) else (
    echo ❌ resnet50.pt not found - run 'git lfs pull'
    exit /b 1
)

if exist "gaze-estimation-testing-main\gaze-estimation\weights\mobileone_s0_gaze.onnx" (
    echo ✅ mobileone_s0_gaze.onnx found
) else (
    echo ⚠️  mobileone_s0_gaze.onnx not found
)

echo.
echo 🔨 Building Docker image...
docker-compose build

if %errorlevel% equ 0 (
    echo.
    echo ✅ Docker image built successfully!
    echo.
    echo To start the API, run:
    echo   docker-compose up
    echo.
    echo Or run in background:
    echo   docker-compose up -d
) else (
    echo.
    echo ❌ Build failed! Check errors above.
    exit /b 1
)

