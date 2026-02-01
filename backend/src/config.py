import os
import json
import time
from pathlib import Path

LABEL_MAP = {
    "NormalVideos": 0,
    "Abuse": 1,
    "Arrest": 1,
    "Arson": 1,
    "Assault": 1,
    "Burglary": 1,
    "Explosion": 1,
    "Fighting": 1,
    "RoadAccidents": 1,
    "Robbery": 1,
    "Shooting": 1,
    "Shoplifting": 1,
    "Stealing": 1,
    "Vandalism": 1
}


POSE_MODEL_PATH = "yolo/yolov8n-pose.pt"
BEHAVIOR_MODEL_PATH = "models/behavior_xgb.pkl"


RISK_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.5

SEQUENCE_LENGTH = 30  
FPS = 5  


ALERT_COOLDOWN = 5  # Seconds between alerts for the same person

# Shared alert data file for communication between detection and dashboard
PROJECT_DIR = Path(__file__).parent.parent
ALERTS_FILE = PROJECT_DIR / "alerts_data.json"

def get_camera_label(video_source):
    """Get a human-readable label for the video source."""
    if isinstance(video_source, int):
        return f"Camera {video_source}"
    elif isinstance(video_source, str):
        if video_source.isdigit():
            return f"Camera {video_source}"
        else:
            return os.path.basename(video_source)
    return "Unknown Source"

def save_alert(alert_data):
    """Save an alert to the shared alerts file."""
    try:
        alerts = load_alerts()
        alerts["alerts"].append(alert_data)
        # Keep only the last 100 alerts
        alerts["alerts"] = alerts["alerts"][-100:]
        alerts["last_updated"] = time.time()
        alerts["is_threat_active"] = True
        alerts["current_threat"] = alert_data
        alerts["total_alert_count"] = len(alerts["alerts"])  # Add total count for dashboard
        
        with open(ALERTS_FILE, 'w') as f:
            json.dump(alerts, f, indent=2)
        
        print(f"[ALERT SAVED] Total alerts now: {len(alerts['alerts'])} | File: {ALERTS_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save alert: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_alerts():
    """Load alerts from the shared alerts file."""
    try:
        if ALERTS_FILE.exists():
            with open(ALERTS_FILE, 'r') as f:
                data = json.load(f)
                # Ensure total_alert_count is present
                if "total_alert_count" not in data:
                    data["total_alert_count"] = len(data.get("alerts", []))
                return data
    except Exception as e:
        print(f"Error loading alerts: {e}")
    
    return {
        "alerts": [],
        "last_updated": 0,
        "is_threat_active": False,
        "current_threat": None,
        "camera_source": None,
        "total_alert_count": 0
    }

def clear_alerts():
    """Clear all alerts."""
    try:
        with open(ALERTS_FILE, 'w') as f:
            json.dump({
                "alerts": [],
                "last_updated": time.time(),
                "is_threat_active": False,
                "current_threat": None,
                "camera_source": None
            }, f, indent=2)
    except Exception as e:
        print(f"Error clearing alerts: {e}")

def update_detection_status(is_running, camera_source=None, threat_level=0.0):
    """Update the detection status in the shared file."""
    try:
        alerts = load_alerts()
        alerts["is_running"] = is_running
        alerts["camera_source"] = get_camera_label(camera_source) if camera_source is not None else alerts.get("camera_source")
        alerts["threat_level"] = threat_level
        alerts["last_updated"] = time.time()
        
        # Clear threat if not running
        if not is_running:
            alerts["is_threat_active"] = False
            alerts["current_threat"] = None
        
        with open(ALERTS_FILE, 'w') as f:
            json.dump(alerts, f, indent=2)
    except Exception as e:
        print(f"Error updating detection status: {e}")
