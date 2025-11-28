# MogFace & Eye Contact Detection

This project leverages a high-performance **MogFace** model for robust face detection, integrated with an eye contact analysis pipeline.

## MogFace Implementation
The face detection module uses a custom implementation of MogFace featuring:
- **Backbone**: ResNet152 (Deep Residual Network)
- **Architecture**: LFPN (Large Field-of-View Pyramid Network) with DSFD (Dual Shot Face Detector) and CPM (Context Pyramid Module) blocks.
- **Performance**: Optimized for high accuracy in diverse conditions.

## Eye Contact Estimation
The eye contact analysis module utilizes a specialized CNN:
- **Backbone**: ResNet50
- **Input**: Aligned face crops
- **Output**: Eye contact probability score


## Usage
Run the main script with the `--help` flag to see all available arguments and options:

```bash
python main.py --help
```

### Example
```bash
python main.py input_video.mp4 -o output_video.mp4
```
