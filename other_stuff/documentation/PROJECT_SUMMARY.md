# Gaze Estimation Project - Technical Summary

## Overview

This project implements **real-time gaze estimation** on video files using deep learning models. It processes videos frame-by-frame to detect faces and predict where people are looking, outputting annotated videos with gaze direction arrows.

## How It Works

### 1. **Face Detection** (First Stage)
- Uses **RetinaFace** (MobileNet V2 backbone) via the `uniface` library
- Detects faces in each video frame
- Outputs bounding boxes and facial keypoints
- Model: `RetinaFaceWeights.MNET_V2` (pre-downloaded to `~/.uniface/models/`)

### 2. **Gaze Estimation** (Second Stage)
For each detected face:

**a) Face Preprocessing:**
   - Crops face region from bounding box
   - Converts BGR → RGB
   - Resizes to 448×448 pixels
   - Normalizes with ImageNet mean/std values

**b) Model Inference:**
   - Passes preprocessed face through gaze estimation model
   - Model outputs two predictions:
     - **Pitch** (vertical angle): up/down direction
     - **Yaw** (horizontal angle): left/right direction
   
**c) Angle Prediction:**
   - Model uses **binned classification** approach:
     - For Gaze360 dataset: 90 bins, 4° width each, range: -180° to +180°
     - Outputs are softmax probabilities across bins
     - Final angle = weighted sum of bin centers
   
**d) Visualization:**
   - Draws bounding box around face
   - Draws arrow from face center showing gaze direction
   - Arrow direction calculated from pitch/yaw angles

### 3. **Video Processing Pipeline**
```
Input Video → Frame Extraction → Face Detection → 
For each face: Crop → Preprocess → Gaze Estimation → 
Draw Annotations → Output Video
```

## Models Available

### Primary Model (Currently Used)
**ResNet-50**
- **Architecture**: Deep Residual Network with Bottleneck blocks [3, 4, 6, 3 layers]
- **Size**: 91.3 MB
- **Accuracy**: MAE 11.34° (Mean Absolute Error)
- **Training**: 200 epochs on Gaze360 dataset
- **Output**: Two fully connected heads (fc_yaw, fc_pitch), each with 90 bins
- **Weight File**: `gaze-estimation/weights/resnet50.pt`

### Alternative Models Supported
1. **ResNet-18** - 43 MB, MAE: 12.84°
2. **ResNet-34** - 81.6 MB, MAE: 11.33°
3. **MobileNet V2** - 9.59 MB, MAE: 13.07° (lightweight, faster)
4. **MobileOne S0** - 4.8 MB, MAE: 12.58° (ultra-fast inference)
5. **MobileOne S1-S4** - Available but weights not included

## Technical Details

### Model Architecture (ResNet-50)
```
Input: 448×448×3 RGB image
↓
Conv1 (7×7, stride 2) → BatchNorm → ReLU → MaxPool
↓
ResNet Layers 1-4 (with Bottleneck blocks)
↓
AdaptiveAvgPool → Flatten
↓
├─ FC Layer (Yaw) → 90 bins
└─ FC Layer (Pitch) → 90 bins
```

### Key Features
- **Pre-trained backbone**: Uses ImageNet pre-trained weights, then fine-tuned on Gaze360
- **Binned classification**: Treats gaze estimation as classification problem (more stable than direct regression)
- **GPU acceleration**: Automatically uses CUDA if available (`torch.cuda.is_available()`)
- **Dataset**: Trained on Gaze360 (large-scale gaze dataset with 360° coverage)

### Dependencies
- **PyTorch**: Deep learning framework
- **OpenCV**: Video I/O and visualization
- **uniface**: Face detection (RetinaFace)
- **torchvision**: Image transformations

## Your Current Setup

**Command Run:**
```bash
python run_gaze_estimation.py --source input/IMG_1547.MOV --output output/IMG_1547.mp4
```

**Configuration:**
- **Model**: ResNet-50
- **Dataset config**: Gaze360 (90 bins, 4° binwidth, ±180° range)
- **Input**: `input/IMG_1547.MOV`
- **Output**: `output/IMG_1547.mp4`
- **Device**: CPU (should be GPU - see PyTorch CUDA setup)

## Performance Characteristics

### ResNet-50 Model
- **Accuracy**: 11.34° average angular error
- **Speed**: 
  - CPU: ~2-5 FPS (slow)
  - GPU (RTX 4070): ~30-60 FPS (fast) *after installing CUDA-enabled PyTorch*
- **Memory**: ~500-800 MB GPU memory

### Trade-offs
- **ResNet-50**: Best accuracy, moderate speed
- **MobileNet V2**: Smaller, faster, slightly less accurate (13.07°)
- **MobileOne S0**: Smallest, fastest, good accuracy (12.58°)

## Workflow Summary

1. **Setup** (`setup.py`): Installs dependencies
2. **Weights**: Download pre-trained ResNet-50 weights (91.3 MB)
3. **Inference** (`run_gaze_estimation.py`):
   - Wrapper script that calls `gaze-estimation/inference.py`
   - Handles path resolution and argument passing
4. **Processing**:
   - Opens video with OpenCV
   - Per frame: Detect faces → Estimate gaze → Draw annotations
   - Saves annotated video to output path

## Output

The output video contains:
- Original video frames
- Green bounding boxes around detected faces
- Red arrows showing gaze direction
- Arrow originates from face center
- Arrow points in the estimated gaze direction

## Citation

Based on **MobileGaze** (Valikhujaev, 2024), built on top of **L2CS-Net**.
- GitHub: https://github.com/yakhyo/gaze-estimation
- DOI: 10.5281/zenodo.14257640

