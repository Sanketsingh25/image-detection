# 🎯 Real-Time Ball Detection & Tracking with Arduino Control

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/VGG16-Transfer%20Learning-blue" />
  <img src="https://img.shields.io/badge/OpenCV-4.8-green?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/Arduino-Serial%20Control-teal?logo=arduino" />
  <img src="https://img.shields.io/badge/Python-3.x-yellow?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Albumentations-Augmentation-purple" />
</p>

> A real-time computer vision system that detects a ball using a custom-trained VGG16-based object detector, estimates its distance and angle from the camera center, and sends directional commands to an Arduino over serial — enabling a physical system to autonomously track and follow the ball.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset & Augmentation](#-dataset--augmentation)
- [Model Architecture](#-model-architecture)
- [Training Results](#-training-results)
- [Arduino Command Protocol](#-arduino-command-protocol)
- [Getting Started](#-getting-started)
- [Author](#-author)

---

## 🧭 Overview

This project builds a **complete ball tracking pipeline** from scratch:

1. **Data collection** — Custom image dataset with LabelMe JSON annotations
2. **Augmentation** — 60× augmentation per image using Albumentations (crop, flip, brightness, gamma, RGB shift)
3. **Model training** — VGG16 backbone fine-tuned for simultaneous **classification** (ball present/absent) and **bounding box regression**
4. **Real-time inference** — Webcam feed processed at runtime with live bounding box overlay
5. **Physical control** — Angle and distance computed from bbox center → Arduino serial commands for directional movement

The result is a closed-loop system where a physical device (robot/turret) autonomously orients itself toward a detected ball.


Sample test predictions from the trained model:

| Detection confidence > 0.9 | Bounding box drawn in real-time |
|---|---|
| Ball present → bbox + distance overlay | Ball absent → no rectangle drawn |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Live Webcam Feed                        │
│                     (800×800 crop)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              VGG16-Based Ball Detector                      │
│   Input: 120×120 RGB  │  Output: [class_prob, bbox(4)]     │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  ┌───────────────┐       ┌──────────────────────┐
  │  Distance     │       │   Angle from Center  │
  │  Estimation   │       │   (arctan2 of bbox   │
  │  (focal len.) │       │    center vector)    │
  └───────┬───────┘       └──────────┬───────────┘
          │                          │
          └────────────┬─────────────┘
                       │
                       ▼
          ┌────────────────────────┐
          │   Arduino Serial CMD   │
          │  [0]=Stop [1]=Back     │
          │  [2]=Fwd  [3]=Far      │
          │  [4]=Close             │
          └────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🎯 **Ball Detection** | VGG16-based single-object detector (classification + regression) |
| 📦 **Custom Dataset** | LabelMe JSON annotations, manually labeled with bounding boxes |
| 🔁 **Data Augmentation** | 60× per image — crop, flip, brightness, gamma, RGB shift |
| 📐 **Distance Estimation** | Focal length + perceived size formula for depth approximation |
| 🧭 **Angle Computation** | `arctan2` of bbox center vector from frame center |
| 🤖 **Arduino Control** | 5-command serial protocol for directional movement |
| 📊 **Training Monitoring** | TensorBoard callbacks, loss curves (total / class / regress) |
| 💾 **Model Persistence** | Saved and reloaded as `balltracker.h5` |

---

## 🛠️ Tech Stack

- **Deep Learning:** TensorFlow 2.x, Keras
- **Backbone:** VGG16 (pretrained on ImageNet, `include_top=False`)
- **Computer Vision:** OpenCV 4.8
- **Augmentation:** Albumentations
- **Hardware Interface:** cvzone `SerialObject` → Arduino on `COM3`
- **Annotation Tool:** LabelMe (JSON format)
- **Visualization:** Matplotlib, TensorBoard

---

## 📁 Project Structure

```
ball_detection/
├── training_code.ipynb       # Full training pipeline (data → model)
├── ball_detection.py         # Real-time inference + Arduino control
├── balltracker.h5            # Trained model weights
├── logs/                     # TensorBoard training logs
├── data/
│   ├── images/               # Raw labeled images
│   ├── labels/               # LabelMe JSON annotations
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── val/
│       ├── images/
│       └── labels/
└── agm_data/                 # Augmented dataset (60× expansion)
    ├── train/
    ├── test/
    └── val/
```

---

## 📊 Dataset & Augmentation

- **Raw images:** ~22 labeled images (train/test/val split: 70/15/15)
- **Annotation format:** LabelMe JSON with `points` as `[[x1,y1],[x2,y2]]` bounding box
- **After augmentation:** 1320 train / 180 test / 300 val samples

**Augmentation pipeline (Albumentations):**
```python
alb.Compose([
    alb.RandomCrop(width=800, height=800),
    alb.HorizontalFlip(p=0.5),
    alb.RandomBrightnessContrast(p=0.2),
    alb.RandomGamma(p=0.2),
    alb.RGBShift(p=0.2),
    alb.VerticalFlip(p=0.5)
], bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))
```

Images are resized to **120×120** and normalized to `[0, 1]` before training.

---

## 🧠 Model Architecture

Custom dual-head model built on top of VGG16:

```
Input (120×120×3)
    │
VGG16 (pretrained, no top) → (None, None, None, 512)
    │
    ├── GlobalMaxPooling2D → Dense(2048, relu) → Dense(1, sigmoid)
    │                                              [Classification head]
    │
    └── GlobalMaxPooling2D → Dense(2048, relu) → Dense(4, sigmoid)
                                                   [Regression head]

Total params: 16,826,181
```

**Custom loss function:**
```python
# Combined loss = localization_loss + 0.5 * binary_crossentropy
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
    delta_size  = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true - h_pred))
    return delta_coord + delta_size
```

**Optimizer:** Adam with Exponential Decay LR (`initial_lr=0.0001`, `decay_rate=0.96`)

---

## 📈 Training Results

Trained for **10 epochs** on augmented dataset (batch size 8):

| Epoch | Train Loss | Val Loss |
|-------|-----------|---------|
| 1 | 0.3526 | 0.4270 |
| 3 | 0.0193 | 0.1966 |
| 5 | 0.0109 | 0.3496 |
| 7 | 0.0108 | 0.0905 |
| 10 | **0.0087** | **0.1510** |

Classification loss converged to near zero rapidly, with regression loss as the dominant training signal — as expected for single-class detection.

---

## 🤖 Arduino Command Protocol

The inference script computes the **angle** (arctan2 from frame center to bbox center) and **distance** (focal length estimation), then sends one of 5 commands via serial:

| Command | Byte Sent | Condition |
|---------|-----------|-----------|
| Stop / Idle | `[0]` | Ball centered and at correct distance |
| Reverse / Turn Back | `[1]` | Angle < -110° or > 110° |
| Forward | `[2]` | Angle within ±70° of center |
| Move Away (too far) | `[3]` | Distance > 70 cm |
| Move Closer (too close) | `[4]` | Distance < 60 cm |

```python
# Distance formula
distance = (real_object_size * focal_length) / perceived_size
# real_object_size = 19 cm, focal_length = 1.50
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow opencv-python albumentations cvzone matplotlib numpy
```

### Training

1. Label images using [LabelMe](https://github.com/labelmeai/labelme) and export as JSON
2. Organize into `data/train/`, `data/test/`, `data/val/` with `images/` and `labels/` subfolders
3. Run the augmentation cell in `training_code.ipynb` to generate `agm_data/`
4. Train:
```bash
jupyter notebook training_code.ipynb
```

### Real-Time Inference

Connect Arduino to `COM3`, then:

```bash
python ball_detection.py
```

Press `q` to quit the live window.

### Run with saved model

```python
from tensorflow.keras.models import load_model
balltracker = load_model('balltracker.h5')
```

---

## 👤 Author

**Sanket Singh**
Computer Vision · Robotics · Deep Learning

- 🔗 [GitHub](https://github.com/Sanketsingh25)
- 💼 [LinkedIn](https://linkedin.com/in/your-profile) *(update this)*

---

<p align="center">
  <i>Built from scratch — dataset, augmentation, model, and hardware control.</i>
</p>
