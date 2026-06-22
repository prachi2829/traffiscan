#  TraffiScan: Real-Time Traffic Analytics Dashboard

TraffiScan is a real-time traffic analytics system that leverages **YOLOv8** for vehicle detection and **DeepSORT** for multi-object tracking. The system analyzes traffic videos, assigns unique IDs to vehicles, computes tracking metrics, and visualizes results through an interactive Streamlit dashboard.

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

The Streamlit dashboard enables users to upload videos, run detection and tracking, and visualize traffic analytics in real time.

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
