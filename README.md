# Underwater Trash Detection using YOLO11

![Detection Showcase](result_yolov11.png)

## 📌 Overview
This project focuses on the automated detection of underwater debris using the latest **YOLO11** (You Only Look Once) architecture. By leveraging state-of-the-art computer vision, we aim to assist Marine Research and underwater cleanup operations by identifying trash, marine life, and ROV equipment with higher efficiency and accuracy.

---

## 🚀 Features
- **Next-Gen Detection**: Optimized performance using the YOLO11s architecture.
- **Three-Class Classification**:
    - 🟥 **Trash**: Man-made debris (plastic, metal, glass).
    - 🟩 **Bio**: Marine life and biological features.
    - 🟦 **Rov**: Parts of the underwater vehicle/tools.
- **Improved Efficiency**: Faster training and inference compared to previous versions.

---

## 📊 Training Results
The model was trained for **200 epochs** over **2.219 hours**, showing significant speed improvements while maintaining high accuracy.

### Training Environment
- **GPU**: 2x NVIDIA Tesla T4
- **Framework**: Ultralytics YOLO11
- **Model**: YOLO11s (9.4M parameters)
- **Image Size**: 416x416

### Validation Benchmarks
| Class | Images | Instances | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **all** | 1795 | 2578 | 0.955 | 0.929 | 0.964 | 0.786 |
| **Bio** | 419 | 491 | 0.978 | 0.951 | 0.980 | 0.845 |
| **Rov** | 310 | 400 | 0.928 | 0.925 | 0.959 | 0.828 |
| **Trash** | 1408 | 1687 | 0.961 | 0.910 | 0.954 | 0.686 |

### Performance Curves
![Training Results](training_result/results.png)

### Model Evaluation
| Confusion Matrix | Validation Predictions |
| :---: | :---: |
| ![Confusion Matrix](training_result/confusion_matrix.png) | ![Validation Predictions](training_result/val_batch0_pred.jpg) |

---

## 🛠️ Installation & Usage

### 1. Requirements
```bash
pip install ultralytics roboflow
```

### 2. Running Predictions
To use the trained YOLO11 model for inference:
```python
from ultralytics import YOLO

# Load the model
model = YOLO('training_result/weights/best.pt')

# Predict on an image
results = model.predict(source='result_yolov11.png', save=True, imgsz=416)
```

---

## 📂 Project Structure
- `yolov11.ipynb`: Main notebook covering dataset integration, training, and validation.
- `training_result`: Comprehensive training logs, weights, and evaluation metrics.
- `result_yolov11.png`: Sample showcase of the YOLO11 model predictions.

---

## 🌊 Future of Marine Conservation
By adopting **YOLO11**, this project achieves:
- **Faster Deployment**: 200 epochs completed in just ~2.2 hours.
- **High Precision**: Over 95% mAP50 for detecting trash hotspots.
- **Robustness**: Reliable detection even in complex underwater lighting and clarity conditions.

---
> **Disclaimer**: This project was developed to explore the capabilities of YOLO11 in solving real-world environmental challenges.
