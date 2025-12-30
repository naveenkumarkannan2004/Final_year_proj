# Park Surveillance ML Project

A YOLO-based machine learning system for detecting authorized and unauthorized activities in parks using computer vision.

## 📋 Project Overview

This project uses YOLOv11 (You Only Look Once) to perform real-time object detection for park surveillance. The model is trained to classify activities into two categories:
- **Authorised**: Normal park activities (walking, playing, exercising, etc.)
- **Unauthorised**: Prohibited activities (vandalism, fire-making, spitting, etc.)

### Dataset Information
- **Total Images**: 108 (train: 53, validation: 10, test: 8)
- **Classes**: 2 (Unauthorised, authorised)
- **Image Size**: 512x512 pixels
- **Format**: YOLOv11 format with bounding box annotations

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd d:\park_auth

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Basic training (100 epochs, nano model)
python train.py

# Custom training with specific parameters
python train.py --model n --epochs 50 --batch 8 --img-size 512

# Use larger model for better accuracy (slower training)
python train.py --model s --epochs 100 --batch 16
```

**Model sizes**: `n` (nano - fastest), `s` (small), `m` (medium), `l` (large), `x` (xlarge - most accurate)

### 3. Run Inference

#### On Images
```bash
# Inference on test images
python inference_image.py --model runs/detect/train/weights/best.pt --source dataset/test/images

# Inference on a single image
python inference_image.py --model runs/detect/train/weights/best.pt --source path/to/image.jpg

# Adjust confidence threshold
python inference_image.py --model runs/detect/train/weights/best.pt --source dataset/test/images --conf 0.5
```

#### On Videos
```bash
# Inference on video file
python inference_video.py --model runs/detect/train/weights/best.pt --source path/to/video.mp4

# Display video while processing
python inference_video.py --model runs/detect/train/weights/best.pt --source video.mp4 --display
```

#### Real-time Webcam
```bash
# Real-time surveillance with webcam
python inference_webcam.py --model runs/detect/train/weights/best.pt

# Save frames with unauthorized detections
python inference_webcam.py --model runs/detect/train/weights/best.pt --save-detections

# Use different camera
python inference_webcam.py --model runs/detect/train/weights/best.pt --camera 1
```

## 📁 Project Structure

```
park_auth/
├── dataset/
│   ├── train/          # Training images and labels
│   ├── valid/          # Validation images and labels
│   ├── test/           # Test images and labels
│   └── data.yaml       # Dataset configuration
├── runs/               # Training and inference results
│   ├── detect/train/   # Training outputs (weights, metrics, plots)
│   └── inference/      # Inference results
├── train.py            # Model training script
├── inference_image.py  # Image inference script
├── inference_video.py  # Video inference script
├── inference_webcam.py # Real-time webcam inference
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🎯 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `n` | Model size (n/s/m/l/x) |
| `--epochs` | `100` | Number of training epochs |
| `--batch` | `16` | Batch size |
| `--img-size` | `512` | Input image size |
| `--device` | `auto` | Device (cpu/0/1/etc.) |

## 📊 Model Performance

After training, check the results in `runs/detect/train/`:
- **Confusion Matrix**: `confusion_matrix.png`
- **Training Curves**: `results.png`
- **PR Curve**: `PR_curve.png`
- **F1 Curve**: `F1_curve.png`

Key metrics to monitor:
- **mAP50**: Mean Average Precision at IoU=0.50
- **mAP50-95**: Mean Average Precision at IoU=0.50:0.95
- **Precision**: Ratio of correct positive predictions
- **Recall**: Ratio of actual positives correctly identified

## 🔧 Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Required | Path to trained model (.pt file) |
| `--source` | Required | Image/video/directory path |
| `--conf` | `0.25` | Confidence threshold (0-1) |
| `--iou` | `0.45` | IoU threshold for NMS |
| `--output` | `runs/inference/` | Output directory |

## 💡 Usage Examples

### Example 1: Quick Training Test
```bash
# Train for 5 epochs to test setup
python train.py --epochs 5 --batch 8
```

### Example 2: Production Training
```bash
# Train with medium model for better accuracy
python train.py --model m --epochs 150 --batch 16
```

### Example 3: Batch Image Processing
```bash
# Process all test images with high confidence threshold
python inference_image.py --model runs/detect/train/weights/best.pt --source dataset/test/images --conf 0.5
```

### Example 4: Real-time Monitoring
```bash
# Real-time webcam with alerts and auto-save
python inference_webcam.py --model runs/detect/train/weights/best.pt --save-detections
```

## 🎮 Webcam Controls

When running real-time inference:
- **'q'**: Quit the application
- **'s'**: Save current frame manually
- **'r'**: Reset statistics

## ⚠️ Troubleshooting

### CUDA/GPU Issues
If you encounter GPU errors:
```bash
# Force CPU usage
python train.py --device cpu
```

### Memory Issues
Reduce batch size:
```bash
python train.py --batch 4
```

### Low Detection Accuracy
- Increase training epochs: `--epochs 200`
- Use larger model: `--model m` or `--model l`
- Adjust confidence threshold during inference: `--conf 0.3`

## 📈 Expected Results

With default settings (YOLOv11n, 100 epochs):
- Training time: ~30-60 minutes (GPU) / 2-4 hours (CPU)
- Expected mAP50: 0.70-0.85
- Inference speed: 50-100 FPS (GPU) / 5-10 FPS (CPU)

## 🔍 Dataset Classes

### Unauthorised Activities
- Fire making
- Vandalism
- Spitting
- Property damage
- Other prohibited activities

### Authorised Activities
- Walking
- Playing on playground
- Exercising
- Cycling
- Sitting on benches
- Dog walking
- Family activities

## 📝 Notes

- The model works best with images similar to the training data
- For production deployment, consider using a larger model (s/m/l)
- Adjust confidence threshold based on your use case (higher for fewer false positives)
- Real-time performance depends on hardware capabilities

## 🤝 Contributing

To improve the model:
1. Add more training images
2. Increase data augmentation
3. Fine-tune hyperparameters
4. Use ensemble methods

## 📄 License

Dataset: CC BY 4.0 (from Roboflow Universe)

---

**Built with YOLOv11 by Ultralytics**
