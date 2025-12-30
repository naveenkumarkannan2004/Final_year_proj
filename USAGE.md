# Park Surveillance - Quick Usage Guide

## 🚀 Quick Start - Run Inference

### Method 1: Interactive Mode (Easiest)
```bash
py run_inference.py
```
Then enter:
1. Path to your image or video
2. Confidence threshold (optional, press Enter for default 0.25)

**Example:**
```
Path: dataset/test/images/Fire_Making_Video_Generation_mp4-0002_jpg.rf.bdbc51c34debeae7e97713173f206417.jpg
Confidence: 0.3
```

### Method 2: Command Line with Arguments

#### For Images:
```bash
# Single image
py inference_image.py --source path/to/your/image.jpg

# With custom confidence
py inference_image.py --source path/to/your/image.jpg --conf 0.3

# Directory of images
py inference_image.py --source path/to/images/folder

# Custom output location
py inference_image.py --source path/to/image.jpg --output my_results/images
```

#### For Videos:
```bash
# Single video
py inference_video.py --source path/to/your/video.mp4

# With custom confidence
py inference_video.py --source path/to/video.mp4 --conf 0.3

# Display while processing
py inference_video.py --source path/to/video.mp4 --display

# Custom output location
py inference_video.py --source path/to/video.mp4 --output my_results/videos
```

## 📁 Output Locations

All inference results are saved to:
- **Images**: `inference_results/images/`
- **Videos**: `inference_results/videos/`
- **Interactive mode**: `inference_results/videos/detect/` or `inference_results/images/detect/`

## 🎯 Examples with Test Images

```bash
# Test image 1
py inference_image.py --source dataset/test/images/Fire_Making_Video_Generation_mp4-0002_jpg.rf.bdbc51c34debeae7e97713173f206417.jpg

# Test image 2
py inference_image.py --source dataset/test/images/allowed_activity_10_png.rf.84c4f63903a6a5bd8f40dd48c55b8e1f.jpg

# All test images
py inference_image.py --source dataset/test/images
```

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--source` | Required | Path to image/video/directory |
| `--model` | `runs/detect/train/weights/best.pt` | Your trained model |
| `--conf` | `0.25` | Confidence threshold (0.0-1.0) |
| `--output` | `inference_results/` | Output directory |
| `--no-save` | False | Don't save results |
| `--display` | False | Show video while processing (video only) |

## 🔍 Understanding Results

### Detection Classes:
- **Unauthorised**: Fire making, vandalism, spitting, etc.
- **authorised**: Walking, playing, exercising, etc.

### Confidence Scores:
- **0.25-0.50**: Low confidence (may be false positives)
- **0.50-0.75**: Medium confidence
- **0.75-1.00**: High confidence (very reliable)

## 📊 Model Performance

Your trained model achieved:
- **mAP50**: 0.589 (59% accuracy at 50% IoU)
- **mAP50-95**: 0.288
- **Precision**: 0.641
- **Recall**: 0.545

Best performance on **Unauthorised** class:
- Precision: 0.761
- Recall: 0.727
- mAP50: 0.805

## 💡 Tips

1. **Higher confidence = fewer false positives**: Use `--conf 0.5` for more reliable detections
2. **Lower confidence = more detections**: Use `--conf 0.2` to catch everything
3. **Batch processing**: Point `--source` to a folder to process multiple images
4. **Check output folder**: Results are automatically saved with `_detected` suffix

## 🎥 Real-time Webcam (Optional)

```bash
py inference_webcam.py
```

Press 'q' to quit, 's' to save frame, 'r' to reset stats.
