import cv2
import numpy as np
import joblib
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from feature_extract import extract_features
from config import RISK_THRESHOLD, ALERT_COOLDOWN


model = joblib.load("models/behavior_xgb.pkl")
scaler = joblib.load("models/scaler.pkl")


pose_model = YOLO("yolo/yolov8n-pose.pt")


prev_keypoints = {}
keypoint_history = defaultdict(lambda: deque(maxlen=5))  # Store last 5 frames
last_alert_time = defaultdict(float)

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]


def alert(person_id, risk_score, frame, bbox=None):
    """Send alert for suspicious behavior"""
    current_time = time.time()
    
    # Check cooldown
    if current_time - last_alert_time[person_id] < ALERT_COOLDOWN:
        return
    
    print(f"\n{'='*50}")
    print(f"🚨 ALERT: Suspicious behavior detected!")
    print(f"Person ID: {person_id}")
    print(f"Risk Score: {risk_score:.2%}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}\n")
    
    # Visual alert on frame
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)
        cv2.putText(frame, f"ALERT: {risk_score:.0%}", 
                   (int(x1), int(y1)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    last_alert_time[person_id] = current_time


def draw_pose_skeleton(frame, keypoints, color=(0, 255, 0)):
    """Draw pose skeleton with connections"""
    if keypoints is None or len(keypoints) == 0:
        return
    
    keypoints = np.array(keypoints)
    
    # Draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            pt1 = tuple(keypoints[start_idx].astype(int))
            pt2 = tuple(keypoints[end_idx].astype(int))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, pt1, pt2, color, 2)
    
    # Draw keypoints
    for point in keypoints:
        if point[0] > 0 and point[1] > 0:
            cv2.circle(frame, tuple(point.astype(int)), 4, (0, 255, 255), -1)


def blur_face(frame, bbox, keypoints=None, scale=1.3):
    """Blur an approximate face region inside the person's bbox.
    If head keypoints are available, use them to center the face crop; otherwise use the top portion of bbox.
    """
    x1, y1, x2, y2 = bbox
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    # Prefer head keypoints if available
    fx1, fy1, fx2, fy2 = x1, y1, x1 + w, y1 + int(0.25 * h)
    try:
        if keypoints is not None and len(keypoints) >= 3:
            head_idxs = [0, 1, 2]
            pts = [keypoints[i] for i in head_idxs if i < len(keypoints) and keypoints[i][0] > 0]
            if len(pts) > 0:
                xs = [int(p[0]) for p in pts]
                ys = [int(p[1]) for p in pts]
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                face_w = int(w * 0.35)
                face_h = int(h * 0.25)
                fx1 = cx - face_w
                fx2 = cx + face_w
                fy1 = cy - face_h
                fy2 = cy + face_h
    except Exception:
        # Fallback to top portion of bbox
        fx1, fy1, fx2, fy2 = x1, y1, x2, y1 + int(0.25 * h)

    # Expand region slightly
    cx = (fx1 + fx2) // 2
    cy = (fy1 + fy2) // 2
    fw = int((fx2 - fx1) * scale)
    fh = int((fy2 - fy1) * scale)

    nx1 = max(0, cx - fw // 2)
    ny1 = max(0, cy - fh // 2)
    nx2 = min(frame.shape[1], cx + fw // 2)
    ny2 = min(frame.shape[0], cy + fh // 2)

    if nx2 <= nx1 or ny2 <= ny1:
        return

    roi = frame[ny1:ny2, nx1:nx2]
    if roi.size == 0:
        return

    # Choose an odd kernel size proportional to region size
    kx = max(1, (nx2 - nx1) // 3)
    ky = max(1, (ny2 - ny1) // 3)
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1

    blurred = cv2.GaussianBlur(roi, (kx, ky), 0)
    frame[ny1:ny2, nx1:nx2] = blurred


def run_live_detection(video_source=0):
    """
    Run live behavior detection on video stream.
    
    Args:
        video_source: 0 for webcam, or path to video file
    """
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    print("Starting CivicGuard Live Detection...")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO pose detection
        results = pose_model(frame, verbose=False)
        
        # Process each detected person
        for person_id, result in enumerate(results[0].boxes):
            # Get bounding box
            box = result.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            
            # Get detection confidence
            conf = float(result.conf[0])
            
            # Get keypoints for this person
            if len(results[0].keypoints) > person_id:
                keypoints = results[0].keypoints[person_id].xy.cpu().numpy()[0]
                
                # Add to history for temporal smoothing
                keypoint_history[person_id].append(keypoints)
                
                # Get previous keypoints for this person
                prev_kp = prev_keypoints.get(person_id, None)
                
                # Extract behavioral features
                features = extract_features(keypoints, prev_kp)
                
                # Scale features using the same scaler from training
                features_scaled = scaler.transform([features])
                
                # Predict risk score with temporal smoothing
                risk_score = model.predict_proba(features_scaled)[0][1]
                
                # Temporal smoothing: average risk over last few frames
                if len(keypoint_history[person_id]) >= 3:
                    recent_risks = []
                    history_list = list(keypoint_history[person_id])
                    for i in range(len(history_list)-1):
                        temp_features = extract_features(history_list[i+1], history_list[i])
                        temp_scaled = scaler.transform([temp_features])
                        temp_risk = model.predict_proba(temp_scaled)[0][1]
                        recent_risks.append(temp_risk)
                    
                    # Weighted average: recent frames have more weight
                    if recent_risks:
                        risk_score = (risk_score * 0.5 + np.mean(recent_risks) * 0.5)
                
                # Draw bounding box - blue color
                box_color = (255, 0, 0)  # Blue

                # Privacy: Blur face for low-risk people
                if risk_score < RISK_THRESHOLD:
                    try:
                        blur_face(frame, (x1, y1, x2, y2), keypoints)
                    except Exception:
                        pass

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw pose skeleton - green color
                draw_pose_skeleton(frame, keypoints, (0, 255, 0))
                
                # Draw label with ID and confidence
                label = f"id:{person_id} person {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), 
                            box_color, -1)
                
                # Draw label text
                cv2.putText(frame, label, 
                           (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw risk score with color coding
                risk_text = f"Risk: {risk_score:.1%}"
                risk_color = (0, 255, 0) if risk_score < 0.3 else (0, 165, 255) if risk_score < RISK_THRESHOLD else (0, 0, 255)
                
                cv2.putText(frame, risk_text,
                           (x1, y2 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 2)
                
                # Check if risk exceeds threshold for alert
                if risk_score > RISK_THRESHOLD:
                    alert(person_id, risk_score, frame, (x1, y1, x2, y2))
                    # Change box color to red for high risk
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # Update previous keypoints
                prev_keypoints[person_id] = keypoints
        
        # Show frame
        cv2.imshow('CivicGuard - Live Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nLive detection stopped.")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run CivicGuard live detection on webcam or video file.")
    parser.add_argument('source', nargs='?', default=None, help='Path to video file. Omit to use webcam (default)')
    args = parser.parse_args()

    if args.source is None:
        print("No video provided — using webcam (device 0).")
        run_live_detection(0)
    else:
        # allow passing numeric camera indices as well
        src = args.source
        if src.isdigit():
            src_val = int(src)
        else:
            if not os.path.exists(src):
                print(f"Error: video file '{src}' not found.")
                raise SystemExit(1)
            src_val = src
        print(f"Using video source: {src}")
        run_live_detection(src_val)