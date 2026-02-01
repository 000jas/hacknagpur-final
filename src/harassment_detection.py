import cv2
import numpy as np
import joblib
import time
from collections import defaultdict, deque
from ultralytics import YOLO
from interaction_features import extract_sequence_features
from config import ALERT_COOLDOWN

# Load sequence-based harassment detection model (v2 - 85% accuracy)
model = joblib.load("models/harassment_detector_v2.pkl")
scaler = joblib.load("models/harassment_scaler_v2.pkl")

# Load YOLO pose model
pose_model = YOLO("yolo/yolov8n-pose.pt")

# Sequence tracking (store last N frames for each person)
SEQUENCE_LENGTH = 10
person_keypoint_sequences = defaultdict(lambda: deque(maxlen=SEQUENCE_LENGTH))
last_alert_time = defaultdict(float)

# YOLO pose keypoint connections for drawing skeleton
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Harassment detection threshold
HARASSMENT_THRESHOLD = 0.75  # Higher threshold for sequence-based detection

# How long to visually keep a red alert box after an alert (seconds)
ALERT_DISPLAY_SECONDS = 3.0
alert_display_until = defaultdict(float)


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


def blur_face(frame, bbox, keypoints=None, scale=1.2):
    """Blur an approximate face region inside the person's bbox.
    Uses a slightly smaller crop (shorter height) and a gentler blur kernel for better visibility/privacy tradeoff.
    If head keypoints are available, use them to center the face crop; otherwise use the top portion of bbox.
    """
    x1, y1, x2, y2 = bbox
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)

    # Prefer head keypoints if available
    fx1, fy1, fx2, fy2 = x1, y1, x1 + w, y1 + int(0.18 * h)
    try:
        if keypoints is not None and len(keypoints) >= 3:
            head_idxs = [0, 1, 2]
            pts = [keypoints[i] for i in head_idxs if i < len(keypoints) and keypoints[i][0] > 0]
            if len(pts) > 0:
                xs = [int(p[0]) for p in pts]
                ys = [int(p[1]) for p in pts]
                cx = int(np.mean(xs))
                cy = int(np.mean(ys))
                # slightly narrower and shorter face box
                face_w = int(w * 0.30)
                face_h = int(h * 0.18)
                fx1 = cx - face_w
                fx2 = cx + face_w
                fy1 = cy - face_h
                fy2 = cy + face_h
    except Exception:
        # Fallback to top portion of bbox (shorter height)
        fx1, fy1, fx2, fy2 = x1, y1, x2, y1 + int(0.18 * h)

    # Expand region slightly (scale is reduced)
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

    # Choose a stronger blur: larger odd kernel proportional to region size
    # Ensure kernel is at least 5 and odd so blur is visibly stronger
    kx = max(5, (nx2 - nx1) // 3)
    ky = max(5, (ny2 - ny1) // 3)
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1

    # Increase sigma for stronger blur but cap to a reasonable value
    sigma = max(1.5, min(12.0, float(max(kx, ky))))
    blurred = cv2.GaussianBlur(roi, (kx, ky), sigma)
    frame[ny1:ny2, nx1:nx2] = blurred


def alert_harassment(person_id, risk_score, frame, bbox=None):
    """Send alert for detected harassment pattern"""
    current_time = time.time()
    
    # Check cooldown
    if current_time - last_alert_time[person_id] < ALERT_COOLDOWN:
        return
    
    print(f"\n{'='*70}")
    print(f"🚨 HARASSMENT PATTERN DETECTED!")
    print(f"   Person ID: {person_id}")
    print(f"   Confidence: {risk_score:.1%}")
    print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Pattern: Sustained suspicious behavior over {SEQUENCE_LENGTH} frames")
    print(f"{'='*70}\n")
    
    # Visual alert on frame
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        # Draw thick red box for harassment detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)
        
        # Add warning label
        warning_text = f"HARASSMENT RISK: {risk_score:.0%}"
        cv2.putText(frame, warning_text,
                   (int(x1), int(y1)-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    
    last_alert_time[person_id] = current_time
    # Keep the visual alert (red box) visible for a short period after alerting
    alert_display_until[person_id] = current_time + ALERT_DISPLAY_SECONDS


def draw_interaction_line(frame, person1_kp, person2_kp, distance, color):
    """Draw line between two people showing interaction"""
    if person1_kp is not None and person2_kp is not None:
        p1_valid = person1_kp[person1_kp[:, 0] > 0]
        p2_valid = person2_kp[person2_kp[:, 0] > 0]
        
        if len(p1_valid) > 0 and len(p2_valid) > 0:
            p1_center = np.mean(p1_valid, axis=0).astype(int)
            p2_center = np.mean(p2_valid, axis=0).astype(int)
            
            cv2.line(frame, tuple(p1_center), tuple(p2_center), color, 2, cv2.LINE_AA)
            
            # Draw distance
            mid_point = ((p1_center + p2_center) // 2).astype(int)
            cv2.putText(frame, f"{distance:.0f}px",
                       tuple(mid_point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def run_harassment_detection(video_source=0):
    """
    Run sequence-based harassment detection on video stream.
    
    Args:
        video_source: 0 for webcam, or path to video file
    """
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        return
    
    print("="*70)
    print("🛡️  CivicGuard - Temporal Harassment Detection System")
    print("="*70)
    print("\n📊 System Configuration:")
    print(f"   - Sequence Length: {SEQUENCE_LENGTH} frames")
    print(f"   - Detection Threshold: {HARASSMENT_THRESHOLD:.0%}")
    print(f"   - Features: Temporal patterns + Interpersonal interactions")
    print(f"   - Privacy: Pose-based analysis (no facial recognition)")
    print("\n⌨️  Press 'q' to quit\n")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # current timestamp used for alert display timing
        current_time = time.time()
        
        # Run YOLO pose detection
        results = pose_model(frame, verbose=False)
        
        # Store current frame keypoints
        current_keypoints = {}
        current_boxes = {}
        
        for person_id, result in enumerate(results[0].boxes):
            if person_id >= 2:  # Only track up to 2 people for interaction
                break
            
            box = result.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            conf = float(result.conf[0])
            
            if len(results[0].keypoints) > person_id:
                keypoints = results[0].keypoints[person_id].xy.cpu().numpy()[0]
                
                # Add to sequence history
                person_keypoint_sequences[person_id].append(keypoints)
                current_keypoints[person_id] = keypoints
                current_boxes[person_id] = (x1, y1, x2, y2, conf)
        
        # Perform sequence-based analysis when we have enough frames
        harassment_scores = {}
        
        for person_id in current_keypoints.keys():
            if len(person_keypoint_sequences[person_id]) >= SEQUENCE_LENGTH // 2:
                # Create sequence dict for this person and potentially others
                sequence_dict = {
                    pid: list(person_keypoint_sequences[pid])
                    for pid in current_keypoints.keys()
                }
                
                # Extract sequence features
                features = extract_sequence_features(sequence_dict, SEQUENCE_LENGTH)
                
                # Scale features
                features_scaled = scaler.transform([features])
                
                # Predict harassment risk
                risk_score = model.predict_proba(features_scaled)[0][1]
                harassment_scores[person_id] = risk_score
        
        # Visualize detections
        for person_id in current_keypoints.keys():
            keypoints = current_keypoints[person_id]
            x1, y1, x2, y2, conf = current_boxes[person_id]
            
            # Get risk score
            risk_score = harassment_scores.get(person_id, 0.0)

            # If an alert was recently raised for this person, keep the box red for visibility
            is_alert_active = current_time < alert_display_until.get(person_id, 0.0)

            if is_alert_active:
                box_color = (0, 0, 255)  # Red - active alert
                skeleton_color = (0, 0, 255)
            elif risk_score >= HARASSMENT_THRESHOLD:
                box_color = (0, 0, 255)  # Red - high risk
                skeleton_color = (0, 0, 255)
                detection_count += 1
            elif risk_score >= 0.5:
                box_color = (0, 165, 255)  # Orange - medium risk
                skeleton_color = (0, 165, 255)
            else:
                box_color = (255, 0, 0)  # Blue - normal
                skeleton_color = (0, 255, 0)
            
            # Draw bounding box
            thickness = 3 if risk_score >= HARASSMENT_THRESHOLD else 2

            # Privacy: Blur face for low-risk people
            if risk_score < HARASSMENT_THRESHOLD:
                try:
                    blur_face(frame, (x1, y1, x2, y2), keypoints)
                except Exception:
                    pass

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            
            # Draw pose skeleton
            draw_pose_skeleton(frame, keypoints, skeleton_color)
            
            # Draw label
            label = f"id:{person_id} conf:{conf:.2f}"
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + 180, y1), box_color, -1)
            cv2.putText(frame, label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw risk score
            risk_text = f"Risk: {risk_score:.1%}"
            risk_color = (0, 255, 0) if risk_score < 0.3 else (0, 165, 255) if risk_score < HARASSMENT_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, risk_text, (x1, y2 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
            
            # Alert if threshold exceeded
            if risk_score >= HARASSMENT_THRESHOLD:
                alert_harassment(person_id, risk_score, frame, (x1, y1, x2, y2))
        
        # Draw interaction indicators
        if len(current_keypoints) >= 2:
            person_ids = list(current_keypoints.keys())[:2]
            p1_kp = current_keypoints[person_ids[0]]
            p2_kp = current_keypoints[person_ids[1]]
            
            # Calculate distance
            p1_center = np.mean(p1_kp[p1_kp[:, 0] > 0], axis=0) if np.any(p1_kp[:, 0] > 0) else None
            p2_center = np.mean(p2_kp[p2_kp[:, 0] > 0], axis=0) if np.any(p2_kp[:, 0] > 0) else None
            
            if p1_center is not None and p2_center is not None:
                distance = np.linalg.norm(p1_center - p2_center)
                
                # Color based on distance (red=close, orange=medium, green=far)
                if distance < 100:
                    line_color = (0, 0, 255)
                elif distance < 200:
                    line_color = (0, 165, 255)
                else:
                    line_color = (0, 255, 0)
                
                draw_interaction_line(frame, p1_kp, p2_kp, distance, line_color)
        
        # Display info panel
        info_y = 30
        cv2.rectangle(frame, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_count}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"People Tracked: {len(current_keypoints)}", (20, info_y+30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Sequence Buffer: {max([len(person_keypoint_sequences[p]) for p in person_keypoint_sequences] + [0])}/{SEQUENCE_LENGTH}",
                   (20, info_y+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Alerts: {detection_count}", (20, info_y+90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if detection_count > 0 else (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('CivicGuard - Harassment Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*70}")
    print(f"📊 Session Summary:")
    print(f"   Total Frames: {frame_count}")
    print(f"   Harassment Alerts: {detection_count}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run CivicGuard harassment detection on webcam or video file.")
    parser.add_argument('source', nargs='?', default=None, help='Path to video file. Omit to use webcam (default)')
    args = parser.parse_args()

    if args.source is None:
        print("No video provided — using webcam (device 0).")
        run_harassment_detection(0)
    else:
        src = args.source
        if src.isdigit():
            src_val = int(src)
        else:
            if not os.path.exists(src):
                print(f"Error: video file '{src}' not found.")
                raise SystemExit(1)
            src_val = src
        print(f"Using video source: {src}")
        run_harassment_detection(src_val)
