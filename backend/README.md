# 🛡️ CivicGuard

**AI-Powered Harassment Detection System**

Transform CCTV cameras into proactive safety systems using pose-based behavioral analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/accuracy-85%25-brightgreen.svg)]()

> **Privacy-First**: No facial recognition. Only behavioral pose analysis.

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [How It Works](#-how-it-works)
- [Usage](#-usage)
- [API Integration](#-api-integration)
- [Model Training](#-model-training)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 What It Does

CivicGuard analyzes live video feeds to detect harassment and suspicious behavior patterns **before they escalate**.

### Detection Capabilities:

- ✅ **Personal Space Invasion** - Detects when someone gets too close
- ✅ **Following Behavior** - Identifies persistent trailing patterns
- ✅ **Loitering** - Flags suspicious lingering near targets
- ✅ **Approach Patterns** - Analyzes approach speed and consistency
- ✅ **Evasion Detection** - Recognizes when someone tries to flee

### Real-World Applications:

- 🏢 Corporate offices and workplaces
- 🏫 Educational institutions
- 🏥 Healthcare facilities
- 🚇 Public transportation hubs
- 🏪 Retail stores and malls

**Key Advantage**: Detects behavioral patterns over time, not just single moments.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam or IP camera
- 4GB RAM minimum (8GB recommended)
- (Optional) NVIDIA GPU for faster processing

### 1-Minute Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CivicGuard.git
cd CivicGuard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run detection!
python src/harassment_detection.py
```

**That's it!** Point your camera and the system will start detecting.

---

## 📦 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/CivicGuard.git
cd CivicGuard
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies include:**

- `ultralytics` - YOLOv8 for pose detection
- `opencv-python` - Video processing
- `xgboost` - Machine learning classifier
- `scikit-learn` - Feature scaling and metrics
- `numpy` - Numerical operations
- `flask` - REST API server (optional)

### Step 4: Download Pre-trained Model

The YOLO pose model will download automatically on first run.

**Manual download (if needed):**

```bash
# Create yolo directory
mkdir -p yolo

# Download YOLOv8n-pose
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt -O yolo/yolov8n-pose.pt
```

### Verify Installation

```bash
python -c "import cv2, ultralytics, xgboost; print('✅ All dependencies installed!')"
```

---

## 🔧 How It Works

### System Architecture

```
┌─────────────┐
│ Camera Feed │
└──────┬──────┘
       │
       ▼
┌────────────────────────────┐
│  YOLO Pose Detection       │
│  (17 keypoints per person) │
└──────┬─────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Track People Over Time    │
│  (10-frame sequences)      │
└──────┬─────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Extract 60 Features       │
│  • 40 Temporal             │
│  • 15 Interaction          │
│  • 5 Harassment Indicators │
└──────┬─────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  XGBoost Classifier        │
│  (85% accuracy)            │
└──────┬─────────────────────┘
       │
       ▼
┌────────────────────────────┐
│  Harassment Alert + Score  │
└────────────────────────────┘
```

### Detailed Pipeline

#### 1️⃣ **Pose Detection (YOLOv8n-pose)**

- Detects people in every frame
- Extracts 17 keypoints per person:
  - Head, shoulders, elbows, wrists
  - Hips, knees, ankles
- Confidence scores for each keypoint
- **Speed**: ~30 FPS on CPU, ~100+ FPS on GPU

#### 2️⃣ **Temporal Tracking**

- Tracks each person across 10 consecutive frames
- Builds movement trajectories
- Calculates velocities and accelerations
- Maintains person identity throughout video
- **Window**: 10 frames (~0.33 seconds at 30 FPS)

#### 3️⃣ **Feature Extraction (60 Features)**

**Temporal Features (40):**

- Movement speed and direction
- Acceleration patterns
- Path trajectory analysis
- Position changes over time
- Body orientation dynamics

**Interaction Features (15):**

- Interpersonal distance
- Facing direction alignment
- Personal space zones (intimate, personal, social)
- Proximity duration
- Mutual gaze indicators

**Harassment Indicators (5):**

- `invasion_score` - Personal space violations
- `following_score` - Persistent trailing
- `loitering_score` - Suspicious lingering
- `approach_score` - Aggressive approach patterns
- `evasion_score` - Target fleeing behavior

#### 4️⃣ **Classification (XGBoost)**

- **Model**: XGBoost with 500 estimators
- **Training**: UCF Crime Dataset (Abuse, Assault, Fighting vs Normal)
- **Features**: 60 engineered behavioral features
- **Output**: Harassment probability (0-100%)
- **Threshold**: 70% default (configurable)

#### 5️⃣ **Alert Generation**

- **Standalone Mode** (`harassment_detection.py`): Real-time alerts with full visualization
- **API Mode** (`api_server.py`): Same visualization + aggregated JSON alerts every 20 seconds
- **Output**: JSON with confidence, threat level, session details

---

## 🎮 Usage

### Basic Detection (Webcam)

```bash
python src/harassment_detection.py
```

**Controls:**

- Press `q` to quit
- Real-time visualization with:
  - Person IDs
  - Pose skeletons
  - Confidence scores
  - Alert levels

### REST API Server (For Frontend Integration)

```bash
python src/api_server.py
```

**Features:**

- Built with **FastAPI** (async, modern, fast)
- Uses the same visualization as `harassment_detection.py` (keypoint skeletons, interaction lines)
- Shows live camera feed with all the nice pose drawings
- Aggregates alerts every 20 seconds (configurable) for frontend dashboards
- Prevents frontend spam while maintaining real-time monitoring
- **Auto-generated API docs** at `/docs` (Swagger) and `/redoc`

**API runs on:** `http://localhost:5001`

**API Documentation:**

- Swagger UI: http://localhost:5001/docs
- ReDoc: http://localhost:5001/redoc

**Output Example (API Response):**

```json
{
  "status": "THREAT_DETECTED",
  "threat_level": "HIGH",
  "confidence": 0.87,
  "detections": 15,
  "timestamp": "2026-02-01T14:30:45"
}
```

**Quick test:**

```bash
# Start detection
curl -X POST http://localhost:5001/api/start \
  -H "Content-Type: application/json" \
  -d '{"alert_interval": 20, "confidence_threshold": 0.70}'

# Get latest alert
curl http://localhost:5001/api/latest-alert

# Get statistics
curl http://localhost:5001/api/stats
```

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for full API reference.

---

## 🌐 API Integration

### For Frontend Developers

**Simple polling example (JavaScript/React):**

```javascript
// Start detection
await fetch("http://localhost:5001/api/start", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ alert_interval: 20, confidence_threshold: 0.7 }),
});

// Poll every 5 seconds for alerts
setInterval(async () => {
  const res = await fetch("http://localhost:5001/api/latest-alert");
  const alert = await res.json();

  if (alert.status === "THREAT_DETECTED") {
    console.log(`⚠️  ${alert.threat_level}: ${alert.confidence}%`);
    updateDashboard(alert);
  }
}, 5000);
```

**Threat Levels:**

- `CRITICAL`: ≥ 90% - Immediate response needed
- `HIGH`: 80-89% - Alert security team
- `MEDIUM`: 70-79% - Monitor closely
- `LOW`: < 70% - Log for review

---

## 🎓 Model Training

### Using Pre-trained Model (Recommended)

The repository includes a pre-trained model with **85% accuracy**.

**Location:** `models/harassment_detector_v2.pkl`

### Training Your Own Model

**Step 1: Prepare Dataset**

```bash
# Organize videos into categories
data/frames/train/
├── Abuse/          # Harassment examples
├── Assault/        # Aggressive behavior
├── Fighting/       # Physical confrontation
└── NormalVideos/   # Normal interactions
```

**Step 2: Extract Sequences**

```bash
python src/prepare_harassment_sequences.py
```

This creates:

- Temporal sequences (10-frame windows)
- 60 features per sequence
- Balanced dataset with class weights

**Step 3: Train Model**

```bash
python src/train_harassment_model.py
```

**Output:**

```
Training complete!
Accuracy: 85.03%
Harassment Detection Rate: 93.92%
Models saved to:
  - models/harassment_detector_v2.pkl
  - models/harassment_scaler_v2.pkl
```

**Training Data:**

- Dataset: UCF Crime Dataset
- Sequences: 831 temporal sequences
- Classes: Normal (738) vs Abnormal (93)
- Features: 60 behavioral indicators
- Model: XGBoost (500 estimators, max_depth=12)

---

## 📁 Project Structure

```
CivicGuard/
├── src/
│   ├── harassment_detection.py       # ⭐ Main detection (standalone with visualization)
│   ├── api_server.py                # 🌐 REST API (same visualization + alert aggregation)
│   ├── train_harassment_model.py    # 🎓 Model training
│   ├── prepare_harassment_sequences.py  # 📊 Dataset preparation
│   ├── interaction_features.py      # 🔧 Feature extraction
│   ├── detect_pose.py               # 🎯 YOLO pose utilities
│   └── config.py                    # ⚙️  Configuration
│
├── models/
│   ├── harassment_detector.pkl      # 🧠 Trained model (85% acc)
│   └── sequence_scaler.pkl          # 📏 Feature scaler
│
├── yolo/
│   └── yolov8n-pose.pt             # 👤 YOLO pose model
│
├── data/                            # 📂 UCF Crime Dataset
│   └── frames/train/...
│
├── alerts/                          # 📋 Saved JSON alerts
│
├── README.md                        # 📖 This file
├── API_DOCUMENTATION.md             # 📡 API reference
├── requirements.txt                 # 📦 Dependencies
└── .env                            # 🔐 API keys (optional)
```

---

## 📊 Performance

### Model Metrics

| Metric                        | Value         |
| ----------------------------- | ------------- |
| **Overall Accuracy**          | 85.03%        |
| **Harassment Detection Rate** | 93.92%        |
| **False Alarm Rate**          | 6.08%         |
| **AUC-ROC Score**             | 0.7486        |
| **Processing Speed**          | ~30 FPS (CPU) |

### Top Predictive Features

1. **invasion_score** (7.04%) - Personal space violations
2. **P2_trajectory_x_mean** (4.60%) - Following patterns
3. **max_distance** (4.53%) - Proximity tracking
4. **loitering_score** (3.01%) - Suspicious lingering
5. **approach_speed** (2.88%) - Aggressive movement

### System Requirements

**Minimum:**

- CPU: Intel i5 or equivalent
- RAM: 4GB
- Python: 3.8+
- Camera: 720p

**Recommended:**

- CPU: Intel i7/Ryzen 7 or better
- RAM: 8GB+
- GPU: NVIDIA GTX 1060 or better
- Camera: 1080p+

### Performance Tips

```bash
# Use GPU acceleration (if available)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Reduce sequence length for faster processing
# Edit src/config.py:
SEQUENCE_LENGTH = 5  # Faster but less accurate

# Lower resolution for speed
# In code: resize frames before processing
```

---

## 🛠️ Troubleshooting

### Common Issues

**1. Camera not detected**

```bash
# Test camera
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error')"

# Try different camera indices
python src/harassment_detection.py  # Uses camera 0
# or manually change in code: cv2.VideoCapture(1)
```

**2. Models not found**

```bash
# Verify models exist
ls models/harassment_detector_v2.pkl
ls models/harassment_scaler_v2.pkl

# If missing, retrain:
python src/train_harassment_model.py
```

**3. Low FPS / Slow processing**

```bash
# Install GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Reduce frame resolution
# Edit detection script, add:
# frame = cv2.resize(frame, (640, 480))
```

**4. Too many false alarms**

```bash
# Increase confidence threshold
python src/production_detection.py --threshold 0.85
```

**5. Missing detections**

```bash
# Decrease confidence threshold
python src/production_detection.py --threshold 0.60

# Ensure good lighting
# Check camera angle (front view works best)
```

### Getting Help

1. Check error message carefully
2. Verify all dependencies: `pip list`
3. Ensure Python 3.8+: `python --version`
4. Check camera permissions
5. Review logs in console
6. Open an issue on GitHub with:
   - Error message
   - Python version
   - OS and hardware
   - Steps to reproduce

---

## 🔒 Privacy & Ethics

### Privacy Commitments

✅ **No Facial Recognition** - System only analyzes body poses  
✅ **No Personal Identification** - Tracks behavior, not identity  
✅ **No Data Retention** - Processes in real-time, no storage  
✅ **Transparent Alerts** - Shows confidence scores and reasoning  
✅ **Human Oversight** - Designed to assist, not replace, security personnel

### Ethical Use Guidelines

**DO:**

- Use for safety and security purposes
- Inform people about monitoring
- Combine with human security staff
- Review and validate alerts
- Respect privacy laws and regulations

**DON'T:**

- Use for surveillance without consent
- Rely solely on automated decisions
- Use in private spaces (bathrooms, changing rooms)
- Store video without proper consent
- Use for discrimination or profiling

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**For educational and research purposes.**

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional behavioral patterns
- [ ] Multi-camera tracking
- [ ] Enhanced temporal analysis
- [ ] Mobile app integration
- [ ] Cloud deployment guides
- [ ] Additional dataset support

**To contribute:**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/CivicGuard/issues)
- **Documentation**: [API Docs](API_DOCUMENTATION.md)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/CivicGuard/discussions)

---

## 🙏 Acknowledgments

- **Dataset**: UCF Crime Dataset
- **Pose Detection**: Ultralytics YOLOv8
- **ML Framework**: XGBoost
- **Inspiration**: Creating safer communities through AI

---

## 📈 Roadmap

- [x] Real-time pose-based detection
- [x] Temporal sequence analysis
- [x] REST API for integration
- [x] Production-ready deployment
- [ ] Multi-camera support
- [ ] Cloud deployment
- [ ] Mobile app
- [ ] Alert notifications (SMS/Email)
- [ ] Dashboard UI
- [ ] Advanced analytics

---

**Built with ❤️ for safer communities**

_Making the world safer, one frame at a time._ 🛡️

# System ready!

````

### 2. Run Detection

```bash
# Live webcam detection
python src/harassment_detection.py
````

**That's it!** The system will:

- ✅ Detect people using YOLO pose estimation
- ✅ Track movement patterns over 10-frame sequences
- ✅ Alert when harassment patterns detected
- ✅ Show confidence scores and person IDs

---

## 📊 Performance

- **85% Overall Accuracy**
- **60 Behavioral Features** (temporal + interaction)
- **Trained on UCF Crime Dataset** (Abuse, Assault, Fighting vs Normal)
- **Real-time Processing** (~30 FPS)

---

## 📁 Project Structure

```
CivicGuard/
├── src/
│   ├── harassment_detection.py      # Main detection system ⭐
│   ├── train_harassment_model.py    # Model training
│   ├── prepare_harassment_sequences.py  # Dataset prep
│   ├── interaction_features.py      # Feature extraction
│   └── config.py                    # Settings
├── models/
│   ├── harassment_detector_v2.pkl   # Trained model (85% acc)
│   └── harassment_scaler_v2.pkl     # Feature scaler
├── data/                            # UCF Crime Dataset
└── yolo/
    └── yolov8n-pose.pt             # YOLO pose model
```

---

## 🔧 How It Works

### Detection Pipeline

```
Camera Feed
    ↓
YOLO Pose Detection (17 keypoints/person)
    ↓
10-Frame Temporal Sequences
    ↓
60 Feature Extraction
    ├─ 40 Temporal Features (speed, trajectory, direction)
    ├─ 15 Interaction Features (distance, facing, proximity)
    └─ 5 Harassment Indicators (following, loitering, invasion)
    ↓
XGBoost Classifier (500 estimators)
    ↓
Harassment Score + Alert
```

### Key Features

**Temporal Analysis:**

- Movement trajectories over time
- Speed and acceleration patterns
- Direction changes and path analysis

**Interaction Analysis:**

- Interpersonal distance
- Facing direction alignment
- Personal space invasion
- Proximity zones

**Harassment Indicators:**

- Following behavior detection
- Loitering near target
- Approach consistency
- Evasion attempts

---

## 🎓 Training Your Own Model

### 1. Prepare Dataset

```bash
# Organize videos into categories
data/frames/train/
├── Abuse/
├── Assault/
├── Fighting/
└── NormalVideos/
```

### 2. Extract Sequences

```bash
python src/prepare_harassment_sequences.py
```

### 3. Train Model

```bash
python src/train_harassment_model.py
```

**Output:**

- `models/harassment_detector_v2.pkl` (trained model)
- `models/harassment_scaler_v2.pkl` (feature scaler)
- Performance metrics and feature importance

---

## 🎮 Usage Examples

### Basic Detection

```bash
python src/harassment_detection.py
```

### Custom Configuration

Edit `src/config.py`:

```python
SEQUENCE_LENGTH = 10  # Frames per sequence
FPS = 5              # Processing frame rate
RISK_THRESHOLD = 0.8  # Alert threshold
```

---

## 📈 Model Details

### Architecture

- **Pose Detection**: YOLOv8n-pose (17 keypoints)
- **Classifier**: XGBoost (500 estimators)
- **Features**: 60 temporal + interaction features
- **Window**: 10-frame sequences with stride 5

### Training Data

- **Dataset**: UCF Crime Dataset
- **Categories**: Abuse, Assault, Fighting, NormalVideos
- **Sequences**: 831 temporal sequences
- **Balance**: Weighted class sampling (1:7.94)

### Top Features

1. **invasion_score** (7.04%) - Personal space violations
2. **P2_trajectory_x_mean** (4.60%) - Following patterns
3. **max_distance** (4.53%) - Proximity tracking
4. **loitering_score** (3.01%) - Suspicious lingering
5. **approach_speed** (2.88%) - Aggressive approach

---

## 🔒 Privacy & Ethics

- ✅ **No Facial Recognition** - Only pose analysis
- ✅ **No Personal Identification** - Tracks behavior, not identity
- ✅ **Transparent Alerts** - Shows confidence scores
- ✅ **Human Oversight** - Assists security, doesn't replace them

---

## 🛠️ Requirements

```
Python 3.8+
ultralytics (YOLOv8)
opencv-python
numpy
scikit-learn
xgboost
joblib
```

---

## 📝 License

This project is for educational and research purposes.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional behavioral patterns
- Multi-camera tracking
- Enhanced temporal analysis
- Custom dataset support

---

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Built with ❤️ for safer communities**

made by team 3 musketeers!
