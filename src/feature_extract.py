import numpy as np
import cv2

def extract_features(keypoints, prev_keypoints=None):
    """
    Extract comprehensive behavioral features from pose keypoints.
    
    Args:
        keypoints: Current frame keypoints (17x2 array for YOLO pose)
        prev_keypoints: Previous frame keypoints for motion analysis
    
    Returns:
        List of extracted features (35 features for better accuracy)
    """
    features = []
    
    # Handle empty keypoints
    if keypoints is None or len(keypoints) == 0:
        return [0] * 35
    
    # Ensure keypoints is numpy array
    keypoints = np.array(keypoints)
    
    # Calculate center of mass (average position)
    valid_kp = keypoints[keypoints[:, 0] > 0]
    center = np.mean(valid_kp, axis=0) if len(valid_kp) > 0 else np.array([0, 0])
    
    # === MOTION FEATURES ===
    if prev_keypoints is not None and len(prev_keypoints) > 0:
        prev_keypoints = np.array(prev_keypoints)
        prev_valid = prev_keypoints[prev_keypoints[:, 0] > 0]
        prev_center = np.mean(prev_valid, axis=0) if len(prev_valid) > 0 else center
        
        speed = np.linalg.norm(center - prev_center)
        acceleration = speed * 10
        direction = np.arctan2(center[1] - prev_center[1], center[0] - prev_center[0]) if speed > 0.1 else 0
        
        keypoint_movements = []
        for i in range(min(len(keypoints), len(prev_keypoints))):
            if keypoints[i][0] > 0 and prev_keypoints[i][0] > 0:
                kp_movement = np.linalg.norm(keypoints[i] - prev_keypoints[i])
                keypoint_movements.append(kp_movement)
        
        movement_variance = np.var(keypoint_movements) if keypoint_movements else 0
        max_movement = np.max(keypoint_movements) if keypoint_movements else 0
    else:
        speed = acceleration = direction = movement_variance = max_movement = 0.0
    
    # === POSTURE FEATURES ===
    if len(keypoints) >= 13:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2
        
        torso_vector = shoulder_center - hip_center
        torso_angle = np.arctan2(torso_vector[1], torso_vector[0])
        torso_length = np.linalg.norm(torso_vector)
        body_lean = abs(torso_angle - np.pi/2)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        hip_width = np.linalg.norm(left_hip - right_hip)
    else:
        torso_angle = torso_length = body_lean = shoulder_width = hip_width = 0.0
        shoulder_center = hip_center = np.array([0, 0])
    
    # === SPATIAL FEATURES ===
    if len(valid_kp) > 0:
        x_spread = np.max(valid_kp[:, 0]) - np.min(valid_kp[:, 0])
        y_spread = np.max(valid_kp[:, 1]) - np.min(valid_kp[:, 1])
        aspect_ratio = x_spread / (y_spread + 1e-6)
        compactness = (x_spread * y_spread) / (len(valid_kp) + 1e-6)
    else:
        x_spread = y_spread = aspect_ratio = compactness = 0.0
    
    # === LIMB ANGLE FEATURES ===
    if len(keypoints) >= 17:
        left_arm_angle = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
        right_arm_angle = calculate_angle(keypoints[6], keypoints[8], keypoints[10])
        left_leg_angle = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        right_leg_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
        
        arm_symmetry = abs(left_arm_angle - right_arm_angle)
        leg_symmetry = abs(left_leg_angle - right_leg_angle)
        
        left_elbow_height = keypoints[7][1] - keypoints[5][1] if keypoints[7][0] > 0 and keypoints[5][0] > 0 else 0
        right_elbow_height = keypoints[8][1] - keypoints[6][1] if keypoints[8][0] > 0 and keypoints[6][0] > 0 else 0
        
        left_hand_shoulder_dist = np.linalg.norm(keypoints[9] - keypoints[5]) if keypoints[9][0] > 0 and keypoints[5][0] > 0 else 0
        right_hand_shoulder_dist = np.linalg.norm(keypoints[10] - keypoints[6]) if keypoints[10][0] > 0 and keypoints[6][0] > 0 else 0
    else:
        left_arm_angle = right_arm_angle = left_leg_angle = right_leg_angle = 0.0
        arm_symmetry = leg_symmetry = 0.0
        left_elbow_height = right_elbow_height = 0.0
        left_hand_shoulder_dist = right_hand_shoulder_dist = 0.0
    
    # === HEAD POSITION FEATURES ===
    if len(keypoints) >= 5 and len(shoulder_center) > 0:
        nose = keypoints[0]
        left_eye = keypoints[1]
        right_eye = keypoints[2]
        
        if nose[0] > 0 and shoulder_center[0] > 0:
            head_center = (left_eye + right_eye) / 2 if left_eye[0] > 0 and right_eye[0] > 0 else nose
            head_shoulder_dist = head_center[1] - shoulder_center[1]
        else:
            head_shoulder_dist = 0
    else:
        head_shoulder_dist = 0
    
    # === STANCE FEATURES ===
    if len(keypoints) >= 17:
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        stance_width = np.linalg.norm(left_ankle - right_ankle) if left_ankle[0] > 0 and right_ankle[0] > 0 else 0
        
        left_knee_y = keypoints[13][1]
        left_hip_y = keypoints[11][1]
        crouch_level = (left_hip_y - left_knee_y) if keypoints[13][0] > 0 and keypoints[11][0] > 0 else 0
    else:
        stance_width = crouch_level = 0
    
    # Compile all features (35 features total)
    features.extend([
        speed, acceleration, direction, movement_variance, max_movement,
        torso_angle, torso_length, body_lean, shoulder_width, hip_width,
        x_spread, y_spread, aspect_ratio, compactness,
        left_arm_angle, right_arm_angle, left_leg_angle, right_leg_angle,
        arm_symmetry, leg_symmetry,
        left_elbow_height, right_elbow_height,
        left_hand_shoulder_dist, right_hand_shoulder_dist,
        head_shoulder_dist,
        stance_width, crouch_level,
        center[0] if len(center) > 0 else 0.0,
        center[1] if len(center) > 1 else 0.0,
        float(abs(torso_angle) > 1.0),
        float(speed > 50),
        float(body_lean > 0.5),
        float(movement_variance > 100),
        float(arm_symmetry > 1.5),
        float(crouch_level < -50)
    ])
    
    return features


def calculate_angle(p1, p2, p3):
    """Calculate angle between three points (in radians)"""
    try:
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        return angle
    except:
        return 0.0


def extract_features_from_frame(frame_path, pose_model):
    """
    Extract features from a single frame image.
    
    Args:
        frame_path: Path to the frame image
        pose_model: YOLO pose detection model
    
    Returns:
        Numpy array of features
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        return np.zeros(35)
    
    results = pose_model(frame, verbose=False)
    
    if len(results[0].keypoints) > 0:
        keypoints = results[0].keypoints[0].xy.cpu().numpy()[0]
        features = extract_features(keypoints)
        return np.array(features)
    else:
        return np.zeros(35)
