#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import cv2
import numpy as np
import threading
import time
from datetime import datetime
from collections import defaultdict, deque
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from src.interaction_features import extract_sequence_features

# FastAPI app
app = FastAPI(
    title="CivicGuard Detection API",
    description="AI-Powered Harassment Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
latest_alert = None
alert_history = []
is_running = False
detection_thread = None

# Configuration
MAX_HISTORY = 50

# YOLO pose keypoint connections
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


# Pydantic models for request/response
class StartRequest(BaseModel):
    camera_source: int = 0
    alert_interval: int = 20
    confidence_threshold: float = 0.70


class AlertResponse(BaseModel):
    status: str
    threat_level: Optional[str] = None
    confidence: Optional[float] = None
    avg_confidence: Optional[float] = None
    detections: Optional[int] = None
    timestamp: str
    session_id: Optional[str] = None
    message: Optional[str] = None


class DetectionService:
    """Detection service using original harassment_detection.py logic with visualization."""
    
    def __init__(self, camera_source=0, alert_interval=20, confidence_threshold=0.70):
        # Load models (v2 - 85% accuracy)
        self.model = joblib.load("models/harassment_detector_v2.pkl")
        self.scaler = joblib.load("models/harassment_scaler_v2.pkl")
        self.pose_model = YOLO("yolo/yolov8n-pose.pt")
        
        self.alert_interval = alert_interval
        self.confidence_threshold = confidence_threshold
        self.sequence_length = 10
        
        self.person_keypoint_sequences = defaultdict(lambda: deque(maxlen=self.sequence_length))
        self.last_alert_time = defaultdict(float)
        
        self.camera = cv2.VideoCapture(camera_source)
        self.is_running = False
        self.latest_alert = None
        self.alert_history = []
        
        self.frame_count = 0
        self.detection_count = 0
        self.total_alerts = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.alert_buffer = []
        self.last_api_alert_time = time.time()
    
    def draw_pose_skeleton(self, frame, keypoints, color=(0, 255, 0)):
        if keypoints is None or len(keypoints) == 0:
            return
        
        keypoints = np.array(keypoints)
        
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                pt1 = tuple(keypoints[start_idx].astype(int))
                pt2 = tuple(keypoints[end_idx].astype(int))
                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                    cv2.line(frame, pt1, pt2, color, 2)
        
        for point in keypoints:
            if point[0] > 0 and point[1] > 0:
                cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 255), -1)
    
    def draw_interaction_line(self, frame, person1_kp, person2_kp, distance, color):
        if person1_kp is not None and person2_kp is not None:
            p1_valid = person1_kp[person1_kp[:, 0] > 0]
            p2_valid = person2_kp[person2_kp[:, 0] > 0]
            
            if len(p1_valid) > 0 and len(p2_valid) > 0:
                p1_center = np.mean(p1_valid, axis=0).astype(int)
                p2_center = np.mean(p2_valid, axis=0).astype(int)
                
                cv2.line(frame, tuple(p1_center), tuple(p2_center), color, 2, cv2.LINE_AA)
                
                mid_point = ((p1_center + p2_center) // 2).astype(int)
                cv2.putText(frame, f"{distance:.0f}px",
                           tuple(mid_point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def get_threat_level(self, confidence):
        if confidence >= 0.90:
            return "CRITICAL"
        elif confidence >= 0.80:
            return "HIGH"
        elif confidence >= 0.70:
            return "MEDIUM"
        else:
            return "LOW"
    
    def get_aggregated_alert(self):
        current_time = time.time()
        
        if current_time - self.last_api_alert_time >= self.alert_interval:
            self.last_api_alert_time = current_time
            
            if len(self.alert_buffer) > 0:
                avg_conf = np.mean([a['confidence'] for a in self.alert_buffer])
                max_conf = max([a['confidence'] for a in self.alert_buffer])
                
                alert = {
                    'status': 'THREAT_DETECTED',
                    'threat_level': self.get_threat_level(max_conf),
                    'confidence': float(max_conf),
                    'avg_confidence': float(avg_conf),
                    'detections': len(self.alert_buffer),
                    'timestamp': datetime.now().isoformat(),
                    'session_id': self.session_id
                }
                
                self.alert_buffer = []
                self.total_alerts += 1
                return alert
            else:
                return {
                    'status': 'MONITORING',
                    'threat_level': 'LOW',
                    'confidence': 0.0,
                    'detections': 0,
                    'timestamp': datetime.now().isoformat(),
                    'session_id': self.session_id
                }
        
        return None
    
    def process_frame(self, frame):
        self.frame_count += 1
        
        results = self.pose_model(frame, verbose=False)
        
        current_keypoints = {}
        current_boxes = {}
        
        for person_id, result in enumerate(results[0].boxes):
            if person_id >= 2:
                break
            
            box = result.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            conf = float(result.conf[0])
            
            if len(results[0].keypoints) > person_id:
                keypoints = results[0].keypoints[person_id].xy.cpu().numpy()[0]
                
                self.person_keypoint_sequences[person_id].append(keypoints)
                current_keypoints[person_id] = keypoints
                current_boxes[person_id] = (x1, y1, x2, y2, conf)
        
        harassment_scores = {}
        
        for person_id in current_keypoints.keys():
            if len(self.person_keypoint_sequences[person_id]) >= self.sequence_length // 2:
                sequence_dict = {
                    pid: list(self.person_keypoint_sequences[pid])
                    for pid in current_keypoints.keys()
                }
                
                features = extract_sequence_features(sequence_dict, self.sequence_length)
                features_scaled = self.scaler.transform([features])
                risk_score = self.model.predict_proba(features_scaled)[0][1]
                harassment_scores[person_id] = risk_score
        
        detection = None
        for person_id in current_keypoints.keys():
            keypoints = current_keypoints[person_id]
            x1, y1, x2, y2, conf = current_boxes[person_id]
            
            risk_score = harassment_scores.get(person_id, 0.0)
            
            if risk_score >= self.confidence_threshold:
                box_color = (0, 0, 255)
                skeleton_color = (0, 0, 255)
                self.detection_count += 1
                
                self.alert_buffer.append({
                    'person_id': person_id,
                    'confidence': risk_score,
                    'timestamp': time.time()
                })
                
                detection = {
                    'person_id': person_id,
                    'confidence': float(risk_score),
                    'threat_level': self.get_threat_level(risk_score)
                }
            elif risk_score >= 0.5:
                box_color = (0, 165, 255)
                skeleton_color = (0, 165, 255)
            else:
                box_color = (255, 0, 0)
                skeleton_color = (0, 255, 0)
            
            thickness = 3 if risk_score >= self.confidence_threshold else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            
            self.draw_pose_skeleton(frame, keypoints, skeleton_color)
            
            label = f"id:{person_id} conf:{conf:.2f}"
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 180, y1), box_color, -1)
            cv2.putText(frame, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            risk_text = f"Risk: {risk_score:.1%}"
            risk_color = (0, 255, 0) if risk_score < 0.3 else (0, 165, 255) if risk_score < self.confidence_threshold else (0, 0, 255)
            cv2.putText(frame, risk_text, (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
            
            if risk_score >= self.confidence_threshold:
                warning_text = f"HARASSMENT RISK: {risk_score:.0%}"
                cv2.putText(frame, warning_text,
                           (int(x1), int(y1)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        
        if len(current_keypoints) >= 2:
            person_ids = list(current_keypoints.keys())[:2]
            p1_kp = current_keypoints[person_ids[0]]
            p2_kp = current_keypoints[person_ids[1]]
            
            p1_center = np.mean(p1_kp[p1_kp[:, 0] > 0], axis=0) if np.any(p1_kp[:, 0] > 0) else None
            p2_center = np.mean(p2_kp[p2_kp[:, 0] > 0], axis=0) if np.any(p2_kp[:, 0] > 0) else None
            
            if p1_center is not None and p2_center is not None:
                distance = np.linalg.norm(p1_center - p2_center)
                
                if distance < 100:
                    line_color = (0, 0, 255)
                elif distance < 200:
                    line_color = (0, 165, 255)
                else:
                    line_color = (0, 255, 0)
                
                self.draw_interaction_line(frame, p1_kp, p2_kp, distance, line_color)
        
        info_y = 30
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"People Tracked: {len(current_keypoints)}", (20, info_y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Sequence Buffer: {max([len(self.person_keypoint_sequences[p]) for p in self.person_keypoint_sequences] + [0])}/{self.sequence_length}",
                   (20, info_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Alerts: {self.detection_count}", (20, info_y+90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if self.detection_count > 0 else (255, 255, 255), 2)
        
        return frame, detection
    
    def process_video(self):
        global latest_alert, alert_history, is_running
        
        print("="*70)
        print("🛡️  CivicGuard - Temporal Harassment Detection System (API Mode)")
        print("="*70)
        print(f"\n📊 Configuration:")
        print(f"   - Alert Interval: {self.alert_interval} seconds")
        print(f"   - Threshold: {self.confidence_threshold:.0%}")
        print(f"   - Session ID: {self.session_id}")
        print("\n⌨️  Press 'q' to quit\n")
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            frame, detection = self.process_frame(frame)
            
            alert_report = self.get_aggregated_alert()
            
            if alert_report:
                self.latest_alert = alert_report
                latest_alert = alert_report
                
                self.alert_history.append(alert_report)
                alert_history = self.alert_history[-MAX_HISTORY:]
                
                if alert_report['status'] == 'THREAT_DETECTED':
                    print(f"🚨 [{alert_report['timestamp']}] {alert_report['threat_level']}: {alert_report['confidence']:.1%} ({alert_report['detections']} detections)")
            
            cv2.imshow('CivicGuard - Harassment Detection (API Mode)', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                is_running = False
                break
        
        self.camera.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*70}")
        print(f"📊 Session Summary:")
        print(f"   Total Frames: {self.frame_count}")
        print(f"   Harassment Detections: {self.detection_count}")
        print(f"   API Alerts Sent: {self.total_alerts}")
        print(f"{'='*70}")
    
    def start(self):
        self.is_running = True
        thread = threading.Thread(target=self.process_video, daemon=True)
        thread.start()
        return thread
    
    def stop(self):
        self.is_running = False


# Global service instance
service = None


# ==================== API ENDPOINTS ====================

@app.get("/")
async def index():
    """API documentation root."""
    return {
        "service": "CivicGuard Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "GET /api/health": "Health check",
            "POST /api/start": "Start detection",
            "POST /api/stop": "Stop detection",
            "GET /api/latest-alert": "Get latest alert",
            "GET /api/alert-history": "Get alert history",
            "GET /api/stats": "Get statistics",
            "GET /api/config": "Get configuration",
            "POST /api/clear-history": "Clear alert history"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CivicGuard Detection API",
        "timestamp": datetime.now().isoformat(),
        "detection_active": is_running
    }


@app.post("/api/start")
async def start_detection(config: StartRequest = None):
    """Start detection service."""
    global service, is_running, detection_thread
    
    if is_running:
        raise HTTPException(status_code=400, detail="Detection already running")
    
    if config is None:
        config = StartRequest()
    
    try:
        service = DetectionService(
            camera_source=config.camera_source,
            alert_interval=config.alert_interval,
            confidence_threshold=config.confidence_threshold
        )
        
        detection_thread = service.start()
        is_running = True
        
        return {
            "status": "success",
            "message": "Detection started",
            "session_id": service.session_id,
            "config": {
                "camera_source": config.camera_source,
                "alert_interval": config.alert_interval,
                "confidence_threshold": config.confidence_threshold
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_detection():
    """Stop detection service."""
    global service, is_running
    
    if not is_running:
        raise HTTPException(status_code=400, detail="Detection not running")
    
    try:
        service.stop()
        is_running = False
        
        return {
            "status": "success",
            "message": "Detection stopped",
            "summary": {
                "session_id": service.session_id,
                "total_frames": service.frame_count,
                "total_alerts": service.total_alerts,
                "total_detections": service.detection_count
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/latest-alert")
async def get_latest_alert():
    """Get latest alert (for real-time updates)."""
    global latest_alert
    
    if not is_running:
        raise HTTPException(status_code=400, detail="Detection not running")
    
    if latest_alert is None:
        return {
            "status": "no_alert",
            "message": "No alerts yet",
            "timestamp": datetime.now().isoformat()
        }
    
    return latest_alert


@app.get("/api/alert-history")
async def get_alert_history(limit: int = 10, offset: int = 0, threat_level: str = None):
    """Get alert history."""
    filtered_history = alert_history
    
    if threat_level:
        filtered_history = [
            alert for alert in alert_history
            if alert.get('threat_level') == threat_level.upper()
        ]
    
    paginated = filtered_history[offset:offset+limit]
    
    return {
        "status": "success",
        "total": len(filtered_history),
        "offset": offset,
        "limit": limit,
        "alerts": paginated
    }


@app.get("/api/stats")
async def get_stats():
    """Get detection statistics."""
    if not is_running or service is None:
        raise HTTPException(status_code=400, detail="Detection not running")
    
    total_threats = len([a for a in alert_history if a['status'] == 'THREAT_DETECTED'])
    
    threat_levels = {
        'CRITICAL': len([a for a in alert_history if a.get('threat_level') == 'CRITICAL']),
        'HIGH': len([a for a in alert_history if a.get('threat_level') == 'HIGH']),
        'MEDIUM': len([a for a in alert_history if a.get('threat_level') == 'MEDIUM']),
        'LOW': len([a for a in alert_history if a.get('threat_level') == 'LOW'])
    }
    
    return {
        "status": "success",
        "session_id": service.session_id,
        "stats": {
            "total_frames_processed": service.frame_count,
            "total_alerts": service.total_alerts,
            "total_threats": total_threats,
            "total_detections": service.detection_count,
            "threat_breakdown": threat_levels,
            "detection_rate": f"{(service.detection_count / max(service.frame_count, 1) * 100):.2f}%"
        },
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/config")
async def get_config():
    """Get configuration."""
    if service is None:
        raise HTTPException(status_code=400, detail="No active session")
    
    return {
        "status": "success",
        "config": {
            "alert_interval": service.alert_interval,
            "confidence_threshold": service.confidence_threshold,
            "sequence_length": service.sequence_length
        }
    }


@app.post("/api/clear-history")
async def clear_history():
    """Clear alert history."""
    global alert_history, latest_alert
    
    alert_history = []
    latest_alert = None
    
    return {
        "status": "success",
        "message": "Alert history cleared"
    }


if __name__ == '__main__':
    import uvicorn
    
    print("="*70)
    print("🛡️  CIVICGUARD REST API (FastAPI)")
    print("="*70)
    print("Starting server on http://localhost:5001")
    print("\nAPI Documentation:")
    print("  Swagger UI: http://localhost:5001/docs")
    print("  ReDoc:      http://localhost:5001/redoc")
    print("\nAPI Endpoints:")
    print("  POST /api/start           - Start detection")
    print("  POST /api/stop            - Stop detection")
    print("  GET  /api/latest-alert    - Get latest alert")
    print("  GET  /api/alert-history   - Get alert history")
    print("  GET  /api/stats           - Get statistics")
    print("="*70)
    
    uvicorn.run(app, host="0.0.0.0", port=5001)
