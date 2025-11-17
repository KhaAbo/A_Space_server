#!/usr/bin/env python3
"""
Simple Python script to test the Gaze Estimation API
"""
import time
import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"
TEST_VIDEO = "gaze-estimation-testing-main/input/in_train.mp4"

def test_api():
    """Test the API with a sample video."""
    print("=== Testing Gaze Estimation API ===\n")
    
    # 1. Health check
    print("1. Health Check:")
    try:
        response = requests.get(f"{API_URL}/api/health")
        response.raise_for_status()
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   GPU Available: {health['gpu_available']}")
        print(f"   Model Loaded: {health['model_loaded']}")
        print(f"   Storage Path: {health['storage_path']}")
        print()
    except requests.exceptions.ConnectionError:
        print("   ❌ Error: Cannot connect to API. Is the server running?")
        print("   Start the server with: python -m uvicorn api.main:app")
        sys.exit(1)
    except Exception as e:
        print(f"   ❌ Error: {e}")
        sys.exit(1)
    
    # 2. Upload video
    print("2. Uploading video...")
    if not Path(TEST_VIDEO).exists():
        print(f"   ❌ Error: Test video not found: {TEST_VIDEO}")
        sys.exit(1)
    
    try:
        with open(TEST_VIDEO, "rb") as f:
            response = requests.post(
                f"{API_URL}/api/upload",
                files={"file": (Path(TEST_VIDEO).name, f, "video/mp4")},
                data={"model": "resnet50"}
            )
        response.raise_for_status()
        upload_result = response.json()
        job_id = upload_result["job_id"]
        print(f"   Job ID: {job_id}")
        print(f"   Message: {upload_result['message']}")
        print()
    except Exception as e:
        print(f"   ❌ Error uploading video: {e}")
        sys.exit(1)
    
    # 3. Check status (loop until completed)
    print("3. Checking status...")
    max_attempts = 120  # 10 minutes with 5-second intervals
    attempt = 0
    
    while attempt < max_attempts:
        try:
            response = requests.get(f"{API_URL}/api/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            
            status = job["status"]
            print(f"   Status: {status}", end="")
            
            if status == "processing":
                print(" (processing video...)")
            elif status == "pending":
                print(" (waiting in queue...)")
            else:
                print()
            
            if status in ["completed", "failed"]:
                print(f"\n   Job Details:")
                print(f"   - Filename: {job['filename']}")
                print(f"   - Model: {job['model']}")
                print(f"   - Created: {job['created_at']}")
                print(f"   - Started: {job.get('started_at', 'N/A')}")
                print(f"   - Completed: {job.get('completed_at', 'N/A')}")
                if job.get('error'):
                    print(f"   - Error: {job['error']}")
                print()
                break
            
            time.sleep(5)
            attempt += 1
            
        except Exception as e:
            print(f"\n   ❌ Error checking status: {e}")
            sys.exit(1)
    
    if attempt >= max_attempts:
        print("   ⚠️  Timeout: Processing took too long")
        sys.exit(1)
    
    # 4. Download result (if completed)
    if status == "completed":
        print("4. Downloading result...")
        try:
            response = requests.get(f"{API_URL}/api/download/{job_id}")
            response.raise_for_status()
            
            output_file = "test_output.mp4"
            with open(output_file, "wb") as f:
                f.write(response.content)
            
            file_size_mb = len(response.content) / (1024 * 1024)
            print(f"   ✓ Downloaded to {output_file} ({file_size_mb:.2f} MB)")
            print()
        except Exception as e:
            print(f"   ❌ Error downloading: {e}")
            sys.exit(1)
    else:
        print(f"4. ❌ Processing failed: {job.get('error', 'Unknown error')}")
        sys.exit(1)
    
    print("=== Test Complete ===")
    print("\n✓ All tests passed successfully!")
    print(f"✓ Processed video saved to: test_output.mp4")

if __name__ == "__main__":
    test_api()

