# 🎛 Gesture-Based Volume Control System (Streamlit + OpenCV)

## 📌 Project Overview
This project is a real-time computer vision application that allows users to control system volume using hand gestures captured via a webcam.

The system uses **MediaPipe Hand Tracking** to detect hand landmarks and maps finger distance to volume levels. It provides a touchless, intuitive interface for human-computer interaction.

---

## 🚀 Features

### ✅ Core Features
- Real-time hand detection using webcam
- 21 hand landmarks detection per hand
- Gesture recognition (Mute, Pinch, Open Hand)
- Distance-based volume control
- Smooth volume transition (noise reduction)

### ✅ UI & Visualization
- Live camera feed with overlays
- FPS and latency display
- Detection status (Good / Poor / No Detection)
- Volume bar and gesture indicator
- Distance-to-volume mapping graph
- Real-time volume history graph

### ✅ Advanced Features
- Hand label detection (Left / Right)
- Auto-stop when no hand detected
- Screenshot capture functionality
- Clean and structured Streamlit UI

---

## 🧠 How It Works

1. Webcam captures real-time video
2. MediaPipe detects hand landmarks (21 points)
3. Distance between thumb tip and index finger tip is calculated
4. Distance is mapped to volume (0–100%)
5. Smoothing is applied to avoid sudden jumps
6. UI updates dynamically with graphs and status


