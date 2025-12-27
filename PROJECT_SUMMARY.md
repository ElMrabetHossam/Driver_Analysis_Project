# Driver Behavior Analysis - Project Summary

> A comprehensive driver behavior classification system using the Comma2k19 dataset with computer vision, traditional ML, and deep learning models.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset & Processing](#dataset--processing)
3. [Feature Engineering](#feature-engineering)
4. [Model Performance](#model-performance)
5. [Project Structure](#project-structure)
6. [Usage Guide](#usage-guide)
7. [Results & Insights](#results--insights)

---

## Project Overview

This project implements an end-to-end driver behavior analysis pipeline that:
- Processes dash camera video and vehicle telemetry data
- Extracts visual features using YOLO object detection
- Classifies driving behavior as **safe**, **aggressive**, or **drowsy**
- Computes a driver safety score (0-100)

### Technology Stack
- **Computer Vision**: OpenCV, YOLOv8
- **Traditional ML**: scikit-learn (SVM, Random Forest, K-Means, Isolation Forest)
- **Deep Learning**: PyTorch (LSTM, Transformer)
- **Visualization**: Streamlit, Plotly

---

## Dataset & Processing

### Source Data
- **Dataset**: [Comma2k19](https://github.com/commaai/comma2k19) (~100GB total)
- **Processed**: Chunk 1 (150 out of 188 available segments)
- **Each segment**: ~1 minute of driving with video + sensor data

### Processing Pipeline

```
Comma2k19 Raw Data
       │
       ▼
┌──────────────────────┐
│   Data Loader        │  Loads .hevc video + .npy telemetry
│   (data_loader.py)   │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Feature Extractor   │  YOLO vehicle detection + sensor fusion
│  (feature_extractor) │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Heuristic Labeler   │  Rule-based behavior classification
│  (labeler.py)        │
└──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  training_data.csv   │  35,861 labeled samples
└──────────────────────┘
```

### Data Statistics

| Metric | Value |
|--------|-------|
| Segments processed | 150 |
| Total samples | 35,861 |
| Processing time | ~40 minutes |
| Output file | `data/processed/training_data_150.csv` (7.3 MB) |

### Label Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Aggressive | 22,049 | 61.5% |
| Safe | 12,700 | 35.4% |
| Drowsy | 1,112 | 3.1% |

---

## Feature Engineering

### Extracted Features (18 total)

| Feature | Source | Description |
|---------|--------|-------------|
| `speed` | CAN bus | Vehicle speed (m/s) |
| `steering` | CAN bus | Steering angle (degrees) |
| `accel_forward` | IMU | Forward acceleration (m/s²) |
| `accel_lateral` | IMU | Lateral acceleration (m/s²) |
| `accel_vertical` | IMU | Vertical acceleration (m/s²) |
| `gyro_yaw` | IMU | Yaw rate (rad/s) |
| `radar_distance` | Radar | Distance to lead vehicle (m) |
| `radar_rel_speed` | Radar | Relative speed of lead vehicle (m/s) |
| `vehicle_count` | YOLO | Number of detected vehicles |
| `lead_distance_visual` | YOLO | Visual estimate of lead vehicle distance |
| `speed_change` | Derived | Acceleration from speed (m/s²) |
| `steering_rate` | Derived | Rate of steering change (deg/s) |
| `steering_jerk` | Derived | Rate of steering rate change (deg/s²) |

### Labeling Thresholds

The heuristic labeler classifies behavior based on:

```python
AGGRESSIVE if:
    - steering_jerk > 5.0 deg/s²
    - speed_change < -3.0 m/s² (hard braking)
    - radar_distance < 15m (tailgating)

DROWSY if:
    - lane_deviation > 0.5m (weaving)
    - steering_rate consistently low
    - speed variation minimal

SAFE: otherwise
```

---

## Model Performance

### Final Results (Trained on 35,861 samples)

| Model | Accuracy | F1 Score | Type |
|-------|----------|----------|------|
| SVM | 78.9% | 82.2% | Traditional ML |
| **Random Forest** | **96.6%** | **96.6%** | Traditional ML |
| LSTM | 96.0% | 95.9% | Deep Learning |
| **Transformer** | **96.9%** | **96.6%** | Deep Learning |

> **Best Model**: Transformer with 96.9% accuracy

### Feature Importance (Random Forest)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `steering_jerk` | 34.4% |
| 2 | `speed` | 29.3% |
| 3 | `steering` | 9.1% |
| 4 | `radar_distance` | 5.1% |
| 5 | `steering_rate` | 4.4% |

### Unsupervised Learning Results

**K-Means Clustering** (3 clusters):
- Silhouette Score: 0.178
- Discovered 3 distinct driving styles

**Isolation Forest** (Anomaly Detection):
- Contamination: 10%
- Precision on aggressive driving: 80%

---

## Project Structure

```
Driver_Analysis_Project/
├── data/
│   ├── raw/
│   │   └── comma2k19/
│   │       └── Chunk_1/           # 188 driving segments
│   └── processed/
│       ├── training_data_150.csv  # Main training data
│       ├── demo_annotated.mp4     # Demo video with CV
│       └── demo_resultat.mp4      # Full demo result
│
├── models/
│   ├── random_forest_model.pkl    # Best traditional ML (8.8 MB)
│   ├── svm_model.pkl              # SVM classifier (1.6 MB)
│   ├── lstm_model.pt              # LSTM model (549 KB)
│   └── transformer_model.pt       # Best overall (303 KB)
│
├── src/
│   ├── features/
│   │   ├── data_loader.py         # Comma2k19 data loading
│   │   ├── feature_extractor.py   # YOLO + telemetry extraction
│   │   ├── synchronizer.py        # Time synchronization
│   │   ├── labeler.py             # Heuristic labeling
│   │   ├── batch_process.py       # Multi-segment processing
│   │   └── download_data.py       # Dataset download helper
│   │
│   ├── models/
│   │   ├── traditional_ml.py      # SVM, RF, K-Means, IsoForest
│   │   ├── deep_learning.py       # LSTM, Transformer
│   │   └── scorer.py              # Driver safety scoring
│   │
│   ├── cv/
│   │   ├── lane_detection.py      # Lane detection (Hough)
│   │   └── vehicle_tracking.py    # YOLO vehicle detection
│   │
│   └── dashboard.py               # Streamlit visualization
│
└── requirements.txt               # Python dependencies
```

---

## Usage Guide

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# For deep learning (PyTorch)
pip install torch
```

### ⚠️ Dataset Setup (Required)

The raw dataset is **not included** in this repository due to its size (~8GB). You must download it separately:

1. **Download Chunk_1** from Comma.ai:
   ```bash
   # Option A: Direct download
   wget https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5c5a.torrent
   
   # Option B: Use the provided download script
   python3 src/features/download_data.py --chunk 1
   ```

2. **Place the data** in the correct location:
   ```
   data/
   └── raw/
       └── comma2k19/
           └── Chunk_1/           ← Place downloaded folder here
               ├── b0c9d2329ad1606b|2018-08-02--08-34-47/
               ├── b0c9d2329ad1606b|2018-08-17--14-55-39/
               └── ... (other driving sessions)
   ```

3. **Verify the setup**:
   ```bash
   # Should list available segments
   ls data/raw/comma2k19/Chunk_1/
   
   # Run a quick test
   python3 -c "from src.features.data_loader import Comma2k19Loader; print(Comma2k19Loader('data/raw/comma2k19/Chunk_1').get_available_segments())"
   ```

> **Note**: The processed training data (`data/processed/training_data_150.csv`) is included, so you can use the models and dashboard without downloading the raw data.

### 1. Process More Segments

```bash
# Process all 188 available segments
python3 -m src.features.batch_process \
    --chunk data/raw/comma2k19/Chunk_1 \
    --output data/processed/training_data_full.csv \
    --num-segments 188

# Process with options
python3 -m src.features.batch_process \
    --chunk data/raw/comma2k19/Chunk_1 \
    --output data/processed/output.csv \
    --num-segments 100 \
    --skip-frames 5 \       # Process every 5th frame
    --no-visual             # Skip YOLO (faster)
```

### 2. Train Traditional ML Models

```bash
python3 -m src.models.traditional_ml \
    --train data/processed/training_data_150.csv \
    --output models/

# Output:
# - models/svm_model.pkl
# - models/random_forest_model.pkl
```

### 3. Train Deep Learning Models

```bash
python3 -m src.models.deep_learning \
    --train data/processed/training_data_150.csv \
    --output models/

# Train specific model
python3 -m src.models.deep_learning \
    --train data/processed/training_data_150.csv \
    --model lstm

# Output:
# - models/lstm_model.pt
# - models/transformer_model.pt
```

### 4. Score a Driving Session

```bash
python3 -m src.models.scorer \
    --input data/processed/training_data_150.csv

# Output:
# OVERALL SCORE: 67.1/100
# GRADE: D
# RISK: MODERATE RISK
```

### 5. Launch Dashboard

```bash
python3 -m streamlit run src/dashboard.py

# Opens at http://localhost:8501
```

### 6. Generate Demo Video

```bash
python3 src/main.py
# Creates: data/processed/demo_annotated.mp4
```

---

## Results & Insights

### Key Findings

1. **Steering jerk is the most predictive feature** (34% importance)
   - Abrupt steering changes strongly indicate aggressive driving

2. **Deep learning slightly outperforms traditional ML**
   - Transformer: 96.9% vs Random Forest: 96.6%
   - But Random Forest is faster to train and interpret

3. **Label imbalance exists**
   - 62% aggressive, 35% safe, 3% drowsy
   - Heuristic thresholds may need tuning for production

4. **Highway data has limited lane detection**
   - Lane markings often faded/unclear
   - Radar distance is more reliable for following behavior

### Driver Safety Score Components

| Component | Weight | Description |
|-----------|--------|-------------|
| Behavior | 40% | ML-based classification |
| Smoothness | 25% | Steering/acceleration jerk |
| Awareness | 20% | Following distance |
| Speed | 15% | Speed compliance |

### Sample Score Output

```
========================================
  OVERALL SCORE: 67.1/100
  GRADE: D
  RISK: MODERATE RISK
========================================

--- Score Breakdown ---
  Behavior:   49.0/100
  Smoothness: 58.8/100
  Awareness:  89.2/100
  Speed:      100.0/100

--- Risk Factors ---
  ⚠️  Frequent abrupt steering (60% of time)

--- Recommendations ---
  → Take a break - signs of aggressive driving detected
  → Practice smoother steering and acceleration
```

---

## Contract Decision Framework

Based on the safety score, drivers are automatically classified for contract decisions:

| Score Range | Decision | Action |
|-------------|----------|--------|
| **> 85** | 🟢 **BONUS ELIGIBLE** | Performance bonus, recognition |
| **70-85** | 🔵 **RETAIN CONTRACT** | Standard renewal |
| **50-70** | 🟠 **MANDATORY TRAINING** | Safety training, 30-day review |
| **< 50** | 🔴 **TERMINATE CONTRACT** | HR review, contract termination |

### Generate PDF Report

```bash
# Generate a driver safety report
python3 -m src.models.report_generator \
    --input data/processed/training_data_150.csv \
    --driver-id "DRV_2024_001" \
    --driver-name "John Smith" \
    --output reports/
```

The PDF report includes:
- Executive summary with score and grade
- Contract recommendation with required action items
- Score breakdown with visualizations
- Risk factors and safety recommendations
- Signature lines for management approval

---

## Future Improvements

1. **Process remaining 38 segments** from Chunk 1
2. **Download additional chunks** (Chunks 2-10 available)
3. **Tune labeling thresholds** based on domain expertise
4. **Add real-time inference** from live camera feed
5. **Deploy dashboard** to cloud (Streamlit Cloud, Heroku)

---

## Quick Reference

| Task | Command |
|------|---------|
| Process data | `python3 -m src.features.batch_process --num-segments 188` |
| Train ML | `python3 -m src.models.traditional_ml --train data/processed/training_data_150.csv` |
| Train DL | `python3 -m src.models.deep_learning --train data/processed/training_data_150.csv` |
| Score | `python3 -m src.models.scorer --input data.csv` |
| Dashboard | `python3 -m streamlit run src/dashboard.py` |

