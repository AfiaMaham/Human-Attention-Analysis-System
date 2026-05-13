# Human Attention Analysis System

An AI-based real-time human attention monitoring system developed using **Python, OpenCV, and MediaPipe**.  
The system analyzes eye contact, gaze direction, head pose, blinking behavior, and posture to determine whether a user is attentive or distracted.

---

# Features

- Real-time webcam-based attention monitoring
- Head pose estimation (Pitch, Yaw, Roll)
- Eye contact detection
- Gaze direction tracking
- Blink detection using Eye Aspect Ratio (EAR)
- Slouch/posture detection
- Attention state analysis
- Personalized calibration system
- Real-time visual HUD interface

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

---

# Project Structure

```text
Human_Attention_Analysis_System/
│
├── main.py
├── video_stream.py
│
├── modules/
│   ├── head_pose.py
│   ├── eye_contact.py
│   ├── slouch.py
│   ├── attention_tracker.py
│   └── __init__.py
│
├── utils/
│   ├── draw.py
│   └── __init__.py
│
└── README.md
```

---

# How It Works

The system follows this pipeline:

```text
Webcam Input
     ↓
MediaPipe Landmark Detection
     ↓
Head Pose Estimation
     ↓
Eye & Gaze Analysis
     ↓
Posture Detection
     ↓
Attention Decision Logic
     ↓
HUD Visualization
```

---

# Core Functionalities

## 1. Head Pose Estimation

The system uses OpenCV's `solvePnP()` algorithm with MediaPipe face landmarks to estimate:

- Pitch
- Yaw
- Roll

This determines whether the user is looking left, right, up, or down.

---

## 2. Eye Contact & Gaze Detection

The system tracks:
- Iris landmarks
- Eye corner landmarks

It computes gaze vectors to detect:
- Left
- Right
- Up
- Down
- Forward gaze

---

## 3. Blink Detection

Blinking is detected using the **Eye Aspect Ratio (EAR)** technique.

If EAR falls below a threshold:
- eyes are considered closed,
- blinking state becomes active.

---

## 4. Slouch Detection

Posture analysis is performed using:
- Nose landmark
- Shoulder landmarks

The system calculates neck distance and compares it with a calibrated upright posture baseline.

---

## 5. Attention Analysis

A user is considered inattentive if:
- no eye contact persists for several seconds,
- slouching continues,
- head remains turned away.

The system uses temporal tracking to reduce false positives.

---

# Controls

| Key | Function |
|-----|----------|
| q | Quit application |
| c | Calibrate gaze |
| r | Recalibrate posture |

---

# Applications

- Online learning monitoring
- Workplace productivity systems
- Driver attention monitoring
- Human behavior analysis
- Accessibility systems
- Smart surveillance systems
