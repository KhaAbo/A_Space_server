#!/bin/bash

# API endpoint
API="http://localhost:8000"

echo "=== Testing Gaze Estimation API ==="
echo ""

# 1. Health check
echo "1. Health Check:"
curl -s "$API/api/health" | python -m json.tool
echo ""
echo ""

# 2. Upload video
echo "2. Uploading video..."
RESPONSE=$(curl -s -X POST "$API/api/upload" \
  -F "file=@gaze-estimation-testing-main/input/in_train.mp4" \
  -F "model=resnet50")

echo $RESPONSE | python -m json.tool

# Extract job_id
JOB_ID=$(echo $RESPONSE | python -c "import sys, json; print(json.load(sys.stdin)['job_id'])")
echo ""
echo "Job ID: $JOB_ID"
echo ""

# 3. Check status (loop until completed)
echo "3. Checking status..."
while true; do
    STATUS_RESPONSE=$(curl -s "$API/api/jobs/$JOB_ID")
    STATUS=$(echo $STATUS_RESPONSE | python -c "import sys, json; print(json.load(sys.stdin)['status'])")
    
    echo "Status: $STATUS"
    
    if [ "$STATUS" = "completed" ] || [ "$STATUS" = "failed" ]; then
        echo $STATUS_RESPONSE | python -m json.tool
        break
    fi
    
    sleep 5
done
echo ""

# 4. Download result (if completed)
if [ "$STATUS" = "completed" ]; then
    echo "4. Downloading result..."
    curl -s "$API/api/download/$JOB_ID" -o "test_output.mp4"
    echo "Downloaded to test_output.mp4"
    echo ""
fi

echo "=== Test Complete ==="