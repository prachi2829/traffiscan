# TraffiScan: Real-Time Object Detection and Tracking System

TraffiScan is a Computer Vision-based object detection and tracking system that combines YOLOv8 object detection and DeepSORT multi-object tracking to identify, track, and analyze vehicles in traffic videos. The system assigns persistent IDs to detected objects and visualizes tracking performance through an interactive Streamlit dashboard.

---

## Features

Vehicle Detection using YOLOv8

Multi-Object Tracking using DeepSORT

Traffic Analytics Dashboard

Vehicle ID Assignment

Tracking Performance Metrics

Streamlit-based Interactive Interface

---

## Dashboard Preview

## Dashboard Preview

The Streamlit dashboard enables users to upload videos, perform object detection and tracking, visualize tracked objects, and monitor tracking performance metrics.

![Dashboard](assets/dashboard.png)

---

## Tech Stack

- Python
- OpenCV
- YOLOv8
- DeepSORT
- Streamlit
- NumPy
- Matplotlib

---

## System Pipeline

```text
Input Video
    ↓
YOLOv8 Detection
    ↓
DeepSORT Tracking
    ↓
Vehicle ID Assignment
    ↓
Traffic Analytics
    ↓
Performance Metrics
```

---

## Detection Results

### Sample Detection

![Detection Sample](assets/detection/test.jpg)

### Training Results

![Training Results](assets/detection/results.png)

### Confusion Matrix

![Confusion Matrix](assets/detection/confusion_matrix.png)

---

## Model Performance

| Metric | Score |
|----------|----------|
| Precision | 0.78 |
| Recall | 0.53 |
| mAP@50 | 0.57 |
| mAP@50-95 | 0.32 |

---

## Demo Video

A sample traffic simulation video used for testing is available in:

```text
videos/cardrive.mp4
```

---

## Tracking Performance

![Tracking Metrics](assets/airsim_track/tracking_metrics_graph1.png)

| Metric | Value |
|----------|----------|
| MOTA | 0.996 |
| FPS | 4.20 |
| ID Switches | 32 |
| Avg Track Length | 69.11 |

---

## Project Structure

```text
TraffiScan/
│
├── assets/
│   ├── dashboard.png
│   ├── detection/
│   └── airsim_track/
│
├── models/
│   └── best.pt
│
├── videos/
│   └── cardrive.mp4
│
├── app.py
├── airsim.py
├── training.ipynb
├── data.yaml
└── README.md
```

---

## Running the Project

### Clone Repository

```bash
git clone https://github.com/prachi2829/traffiscan.git
cd traffiscan
```

### Launch Dashboard

```bash
streamlit run app.py
```

---

## Future Enhancements

- Vehicle Counting
- Speed Estimation
- Lane-wise Analytics
- Traffic Congestion Analysis
- Real-Time Camera Support
- Cloud Deployment

---

## Author

**Prachi Yadav**
