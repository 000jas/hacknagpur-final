#!/usr/bin/env python3
"""
CivicGuard - Production-Ready Harassment Detection System
Designed for frontend dashboard integration with aggregated alerts.
"""
import cv2
import numpy as np
import joblib
import os
import json
import time
from collections import defaultdict, deque
from datetime import datetime
from ultralytics import YOLO
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interaction_features import extract_sequence_features

class ProductionHarassmentDetector:
    def __init__(self, 
                 alert_interval=20,  # Report alerts every 20 seconds
                 sequence_length=10,
                 confidence_threshold=0.70):
        """
        Production harassment detection system.
        
        Args:
            alert_interval: Seconds between alert reports (prevents spam)
            sequence_length: Frames in temporal sequence
            confidence_threshold: Minimum confidence for alerts (0-1)
        """
        print("🚀 Initializing Production Harassment Detection System...")
        print(f"   Alert Interval: {alert_interval}s")
        print(f"   Confidence Threshold: {confidence_threshold*100:.0f}%")
        
        # Load models
        self.pose_model = YOLO("yolo/yolov8n-pose.pt")
        
        try:
            self.harassment_model = joblib.load("models/harassment_detector_v2.pkl")
            self.scaler = joblib.load("models/harassment_scaler_v2.pkl")
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
        
        # Configuration
        self.alert_interval = alert_interval
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # State management
        self.person_tracks = defaultdict(lambda: {
            'keypoints_history': deque(maxlen=sequence_length),
            'positions': deque(maxlen=30),
            'last_seen': 0
        })
        
        # Alert aggregation (collect detections over time window)
        self.alert_buffer = []
        self.last_alert_time = time.time()
        self.frame_count = 0
        self.detection_count = 0
        
        # Session tracking
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.total_alerts = 0
        
    def process_frame(self, frame):
        """Process single frame and track detections."""
        self.frame_count += 1
        
        # Detect poses
        results = self.pose_model(frame, verbose=False)
        
        if not results[0].keypoints or len(results[0].keypoints.data) == 0:
            return frame, None
        
        # Extract and track keypoints
        detections = []
        for person_id, keypoints_data in enumerate(results[0].keypoints.data):
            keypoints = keypoints_data.cpu().numpy()
            
            # Calculate center
            valid_kp = keypoints[keypoints[:, 2] > 0.5]
            if len(valid_kp) > 0:
                center = np.mean(valid_kp[:, :2], axis=0)
                
                detections.append({
                    'id': person_id,
                    'keypoints': keypoints,
                    'center': center,
                    'timestamp': time.time()
                })
                
                # Update tracking
                track = self.person_tracks[person_id]
                track['keypoints_history'].append(keypoints)
                track['positions'].append(center)
                track['last_seen'] = self.frame_count
        
        # Analyze pairs for harassment
        harassment_detected = False
        max_confidence = 0
        detection_details = None
        
        if len(detections) >= 2:
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    person1 = detections[i]
                    person2 = detections[j]
                    
                    # Check if we have enough sequence data
                    p1_history = self.person_tracks[person1['id']]['keypoints_history']
                    p2_history = self.person_tracks[person2['id']]['keypoints_history']
                    
                    if len(p1_history) >= self.sequence_length and len(p2_history) >= self.sequence_length:
                        # Build sequence data
                        person_sequences = {
                            0: list(p1_history),
                            1: list(p2_history)
                        }
                        
                        # Extract features
                        features = extract_sequence_features(person_sequences, self.sequence_length)
                        
                        if features is not None:
                            # Scale and predict
                            features_scaled = self.scaler.transform([features])
                            prediction_proba = self.harassment_model.predict_proba(features_scaled)[0]
                            harassment_confidence = prediction_proba[1]
                            
                            if harassment_confidence > max_confidence:
                                max_confidence = harassment_confidence
                            
                            # Check if exceeds threshold
                            if harassment_confidence >= self.confidence_threshold:
                                harassment_detected = True
                                
                                # Calculate distance
                                distance = np.linalg.norm(person1['center'] - person2['center'])
                                
                                detection_details = {
                                    'person1_id': person1['id'],
                                    'person2_id': person2['id'],
                                    'confidence': float(harassment_confidence),
                                    'distance': float(distance),
                                    'timestamp': datetime.now().isoformat(),
                                    'frame_number': self.frame_count
                                }
                                
                                # Add to buffer
                                self.alert_buffer.append(detection_details)
                                
                                # Visual feedback on frame
                                p1_pos = tuple(person1['center'].astype(int))
                                p2_pos = tuple(person2['center'].astype(int))
                                
                                # Color based on confidence
                                if harassment_confidence >= 0.9:
                                    color = (0, 0, 255)  # Red - Critical
                                    level = "CRITICAL"
                                elif harassment_confidence >= 0.8:
                                    color = (0, 140, 255)  # Orange - High
                                    level = "HIGH"
                                else:
                                    color = (0, 255, 255)  # Yellow - Medium
                                    level = "MEDIUM"
                                
                                # Draw connection
                                cv2.line(frame, p1_pos, p2_pos, color, 2)
                                
                                # Draw alert
                                cv2.putText(frame, f"{level}: {harassment_confidence*100:.0f}%",
                                          (p1_pos[0], p1_pos[1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw pose skeletons
        for detection in detections:
            keypoints = detection['keypoints']
            for kp in keypoints:
                if kp[2] > 0.5:
                    cv2.circle(frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
            
            # Person ID
            center = detection['center'].astype(int)
            cv2.putText(frame, f"P{detection['id']}", (center[0], center[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Status overlay
        cv2.putText(frame, f"Frame: {self.frame_count} | People: {len(detections)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if max_confidence > 0:
            cv2.putText(frame, f"Max Threat: {max_confidence*100:.0f}%",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, detection_details
    
    def get_aggregated_alert(self):
        """
        Get aggregated alert report for the time window.
        Returns JSON-ready dict for frontend.
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_alert_time
        
        # Check if interval has passed
        if time_elapsed < self.alert_interval:
            return None
        
        # Generate aggregated report
        if len(self.alert_buffer) == 0:
            # No detections in this window
            self.last_alert_time = current_time
            return {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'interval_seconds': self.alert_interval,
                'status': 'SAFE',
                'detections': 0,
                'max_confidence': 0,
                'alerts': []
            }
        
        # Aggregate detections
        max_confidence = max(d['confidence'] for d in self.alert_buffer)
        avg_confidence = np.mean([d['confidence'] for d in self.alert_buffer])
        
        # Determine overall threat level
        if max_confidence >= 0.9:
            threat_level = "CRITICAL"
        elif max_confidence >= 0.8:
            threat_level = "HIGH"
        elif max_confidence >= 0.7:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        # Build report
        alert_report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'interval_seconds': self.alert_interval,
            'status': 'THREAT_DETECTED',
            'threat_level': threat_level,
            'detections': len(self.alert_buffer),
            'max_confidence': float(max_confidence),
            'avg_confidence': float(avg_confidence),
            'alerts': self.alert_buffer[:10],  # Top 10 detections
            'summary': {
                'total_frames_analyzed': self.frame_count,
                'detection_rate': f"{(len(self.alert_buffer) / self.frame_count * 100):.1f}%"
            }
        }
        
        # Update counters
        self.total_alerts += 1
        self.detection_count += len(self.alert_buffer)
        
        # Clear buffer and reset timer
        self.alert_buffer = []
        self.last_alert_time = current_time
        
        return alert_report
    
    def save_alert_to_file(self, alert_report):
        """Save alert to JSON file for backend integration."""
        if not alert_report:
            return
        
        # Create alerts directory
        os.makedirs("alerts", exist_ok=True)
        
        # Save to file
        filename = f"alerts/alert_{self.session_id}_{self.total_alerts:04d}.json"
        with open(filename, 'w') as f:
            json.dump(alert_report, f, indent=2)
        
        return filename

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Harassment Detection")
    parser.add_argument('--source', type=int, default=0, help='Camera source (0 for webcam)')
    parser.add_argument('--interval', type=int, default=20, help='Alert interval in seconds')
    parser.add_argument('--threshold', type=float, default=0.70, help='Confidence threshold (0-1)')
    parser.add_argument('--save-alerts', action='store_true', help='Save alerts to JSON files')
    args = parser.parse_args()
    
    print("="*70)
    print("🛡️  CIVICGUARD - PRODUCTION HARASSMENT DETECTION")
    print("="*70)
    print(f"Alert Interval: {args.interval}s (prevents frontend spam)")
    print(f"Confidence Threshold: {args.threshold*100:.0f}%")
    print(f"Save Alerts: {'Yes' if args.save_alerts else 'No'}")
    print("="*70)
    
    # Initialize detector
    detector = ProductionHarassmentDetector(
        alert_interval=args.interval,
        confidence_threshold=args.threshold
    )
    
    # Open video source
    cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("\n▶️  Detection started. Press 'q' to quit.\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, detection = detector.process_frame(frame)
            
            # Check for aggregated alert
            alert_report = detector.get_aggregated_alert()
            
            if alert_report:
                print("\n" + "="*70)
                print(f"📊 ALERT REPORT - {alert_report['timestamp']}")
                print("="*70)
                print(f"Status: {alert_report['status']}")
                
                if alert_report['status'] == 'THREAT_DETECTED':
                    print(f"🚨 Threat Level: {alert_report['threat_level']}")
                    print(f"   Detections: {alert_report['detections']}")
                    print(f"   Max Confidence: {alert_report['max_confidence']*100:.1f}%")
                    print(f"   Avg Confidence: {alert_report['avg_confidence']*100:.1f}%")
                    
                    # Show top alerts
                    print(f"\n   Top Alerts:")
                    for i, alert in enumerate(alert_report['alerts'][:3], 1):
                        print(f"   {i}. Person {alert['person1_id']} → Person {alert['person2_id']}: "
                              f"{alert['confidence']*100:.0f}% (Frame {alert['frame_number']})")
                else:
                    print(f"✅ No threats detected in last {args.interval}s")
                
                print("="*70 + "\n")
                
                # Save to file if enabled
                if args.save_alerts:
                    filename = detector.save_alert_to_file(alert_report)
                    if filename:
                        print(f"💾 Alert saved to: {filename}\n")
            
            # Show frame
            cv2.imshow("CivicGuard - Production Detection", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Detection stopped by user")
    
    finally:
        # Final summary
        print("\n" + "="*70)
        print("📈 SESSION SUMMARY")
        print("="*70)
        print(f"Session ID: {detector.session_id}")
        print(f"Total Frames: {detector.frame_count}")
        print(f"Total Alerts: {detector.total_alerts}")
        print(f"Total Detections: {detector.detection_count}")
        print("="*70)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
