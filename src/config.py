
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
