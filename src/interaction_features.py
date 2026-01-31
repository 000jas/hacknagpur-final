import numpy as np
import cv2

def extract_interaction_features(person1_kp, person2_kp):
    """
    Extract features describing interaction between two people.
    
    Args:
        person1_kp: Keypoints of first person
        person2_kp: Keypoints of second person
    
    Returns:
        Interaction features
    """
    if person1_kp is None or person2_kp is None:
        return [0] * 15
    
    p1 = np.array(person1_kp)
    p2 = np.array(person2_kp)
    
    # Centers of mass
    p1_center = np.mean(p1[p1[:, 0] > 0], axis=0) if np.any(p1[:, 0] > 0) else np.array([0, 0])
    p2_center = np.mean(p2[p2[:, 0] > 0], axis=0) if np.any(p2[:, 0] > 0) else np.array([0, 0])
    
    # Interpersonal distance
    distance = np.linalg.norm(p1_center - p2_center)
    
    # Relative positions
    relative_position = p2_center - p1_center
    relative_angle = np.arctan2(relative_position[1], relative_position[0])
    
    # Facing direction (using shoulder orientation)
    if len(p1) >= 7 and len(p2) >= 7:
        p1_shoulder_vec = p1[6] - p1[5] if p1[5][0] > 0 and p1[6][0] > 0 else np.array([1, 0])
        p2_shoulder_vec = p2[6] - p2[5] if p2[5][0] > 0 and p2[6][0] > 0 else np.array([1, 0])
        
        p1_facing = np.arctan2(p1_shoulder_vec[1], p1_shoulder_vec[0])
        p2_facing = np.arctan2(p2_shoulder_vec[1], p2_shoulder_vec[0])
        
        # Are they facing each other?
        facing_alignment = np.cos(p1_facing - p2_facing)
        
        # Is person1 facing person2?
        p1_to_p2_angle = relative_angle
        p1_facing_p2 = np.cos(p1_facing - p1_to_p2_angle)
        
        # Is person2 facing person1?
        p2_to_p1_angle = relative_angle + np.pi
        p2_facing_p1 = np.cos(p2_facing - p2_to_p1_angle)
    else:
        facing_alignment = p1_facing_p2 = p2_facing_p1 = 0
    
    # Body size ratio (height estimation)
    p1_height = np.max(p1[p1[:, 1] > 0, 1]) - np.min(p1[p1[:, 1] > 0, 1]) if np.any(p1[:, 1] > 0) else 0
    p2_height = np.max(p2[p2[:, 1] > 0, 1]) - np.min(p2[p2[:, 1] > 0, 1]) if np.any(p2[:, 1] > 0) else 0
    height_ratio = p1_height / (p2_height + 1e-6)
    
    # Proximity zones (personal space invasion)
    intimate_zone = float(distance < 50)  # Very close
    personal_zone = float(50 <= distance < 150)  # Close
    social_zone = float(150 <= distance < 300)  # Normal
    
    # Relative body orientation
    orientation_diff = abs(p1_facing - p2_facing) if 'p1_facing' in locals() else 0
    
    # Hand positions (defensive/aggressive gestures)
    if len(p1) >= 10 and len(p2) >= 10:
        p1_hands_raised = (p1[9][1] < p1[5][1]) or (p1[10][1] < p1[6][1]) if p1[9][0] > 0 and p1[10][0] > 0 else 0
        p2_hands_raised = (p2[9][1] < p2[5][1]) or (p2[10][1] < p2[6][1]) if p2[9][0] > 0 and p2[10][0] > 0 else 0
    else:
        p1_hands_raised = p2_hands_raised = 0
    
    return [
        distance,
        relative_angle,
        facing_alignment,
        p1_facing_p2,
        p2_facing_p1,
        height_ratio,
        intimate_zone,
        personal_zone,
        social_zone,
        orientation_diff,
        float(p1_hands_raised),
        float(p2_hands_raised),
        relative_position[0],  # X offset
        relative_position[1],  # Y offset
        float(distance < 100 and p1_facing_p2 > 0.5)  # Approaching behavior
    ]


def extract_temporal_features(keypoint_sequence, prev_sequences=None):
    """
    Extract features from a temporal sequence of keypoints.
    
    Args:
        keypoint_sequence: List of keypoint arrays over time (e.g., last 10 frames)
        prev_sequences: Previous sequences for temporal comparison
    
    Returns:
        Temporal behavior features
    """
    if not keypoint_sequence or len(keypoint_sequence) == 0:
        return [0] * 20
    
    features = []
    
    # Calculate trajectory (movement path)
    centers = []
    for kp in keypoint_sequence:
        if kp is not None and len(kp) > 0:
            valid_kp = kp[kp[:, 0] > 0]
            if len(valid_kp) > 0:
                centers.append(np.mean(valid_kp, axis=0))
    
    if len(centers) < 2:
        return [0] * 20
    
    centers = np.array(centers)
    
    # Movement statistics
    displacements = np.diff(centers, axis=0)
    speeds = np.linalg.norm(displacements, axis=1)
    
    avg_speed = np.mean(speeds)
    max_speed = np.max(speeds)
    speed_variance = np.var(speeds)
    
    # Direction changes (erratic movement)
    if len(displacements) > 1:
        angles = np.arctan2(displacements[:, 1], displacements[:, 0])
        angle_changes = np.diff(angles)
        direction_variance = np.var(angle_changes)
        max_direction_change = np.max(np.abs(angle_changes))
    else:
        direction_variance = 0
        max_direction_change = 0
    
    # Path linearity (straight vs. erratic)
    total_displacement = np.linalg.norm(centers[-1] - centers[0])
    path_length = np.sum(speeds)
    linearity = total_displacement / (path_length + 1e-6)
    
    # Acceleration patterns
    if len(speeds) > 1:
        accelerations = np.diff(speeds)
        avg_acceleration = np.mean(accelerations)
        acceleration_variance = np.var(accelerations)
    else:
        avg_acceleration = 0
        acceleration_variance = 0
    
    # Stopping behavior (dwelling/loitering)
    stopped_frames = np.sum(speeds < 5)  # Very slow movement
    stop_ratio = stopped_frames / len(speeds)
    
    # Sudden movements
    sudden_movements = np.sum(speeds > 2 * avg_speed) if avg_speed > 0 else 0
    
    # Posture consistency over time
    posture_changes = 0
    if len(keypoint_sequence) > 1:
        for i in range(len(keypoint_sequence) - 1):
            if keypoint_sequence[i] is not None and keypoint_sequence[i+1] is not None:
                # Compare torso angles
                kp1, kp2 = keypoint_sequence[i], keypoint_sequence[i+1]
                if len(kp1) >= 13 and len(kp2) >= 13:
                    torso1 = kp1[5:7].mean(axis=0) - kp1[11:13].mean(axis=0)
                    torso2 = kp2[5:7].mean(axis=0) - kp2[11:13].mean(axis=0)
                    angle_diff = np.arctan2(torso1[1], torso1[0]) - np.arctan2(torso2[1], torso2[0])
                    if abs(angle_diff) > 0.5:  # Significant posture change
                        posture_changes += 1
    
    # Compile temporal features
    features.extend([
        avg_speed,
        max_speed,
        speed_variance,
        direction_variance,
        max_direction_change,
        linearity,
        avg_acceleration,
        acceleration_variance,
        stop_ratio,
        float(stopped_frames),
        float(sudden_movements),
        float(posture_changes),
        centers[-1][0] - centers[0][0],  # Total X displacement
        centers[-1][1] - centers[0][1],  # Total Y displacement
        path_length,
        float(speed_variance > 50),  # Erratic movement flag
        float(stop_ratio > 0.3),  # Loitering flag
        float(linearity < 0.5),  # Non-linear path flag
        float(avg_speed > 30),  # Fast movement flag
        float(direction_variance > 1.0)  # Frequent direction changes
    ])
    
    return features


def extract_sequence_features(keypoint_sequences, sequence_length=10):
    """
    Extract comprehensive features from multiple person sequences for harassment detection.
    
    Args:
        keypoint_sequences: Dict of {person_id: [kp_frame1, kp_frame2, ...]}
        sequence_length: Number of frames to analyze
    
    Returns:
        Feature vector for sequence-level classification
    """
    features = []
    
    person_ids = list(keypoint_sequences.keys())
    
    # Single person temporal features
    for person_id in person_ids[:2]:  # Analyze up to 2 people
        seq = keypoint_sequences[person_id][-sequence_length:]
        temporal_feats = extract_temporal_features(seq)
        features.extend(temporal_feats)
    
    # Pad if less than 2 people
    while len(features) < 40:  # 2 people * 20 features
        features.extend([0] * 20)
    
    # Interaction features between people
    if len(person_ids) >= 2:
        p1_id, p2_id = person_ids[0], person_ids[1]
        p1_seq = keypoint_sequences[p1_id][-sequence_length:]
        p2_seq = keypoint_sequences[p2_id][-sequence_length:]
        
        # Interaction features over time
        interaction_over_time = []
        for i in range(min(len(p1_seq), len(p2_seq))):
            if p1_seq[i] is not None and p2_seq[i] is not None:
                interaction_feats = extract_interaction_features(p1_seq[i], p2_seq[i])
                interaction_over_time.append(interaction_feats)
        
        if interaction_over_time:
            # Aggregate interaction features
            interaction_arr = np.array(interaction_over_time)
            
            # Distance trends (approaching/retreating)
            distance_trend = np.mean(np.diff(interaction_arr[:, 0])) if len(interaction_arr) > 1 else 0
            
            # Persistent following (person1 consistently facing person2 while moving)
            following_score = np.mean(interaction_arr[:, 3])  # p1_facing_p2
            
            # Personal space violations over time
            invasion_count = np.sum(interaction_arr[:, 6])  # intimate_zone activations
            
            # Mean interaction features
            mean_interaction = np.mean(interaction_arr, axis=0)
            features.extend(mean_interaction.tolist())
            
            # Additional harassment indicators
            features.extend([
                distance_trend,
                following_score,
                float(invasion_count),
                float(following_score > 0.5 and distance_trend < 0),  # Persistent approach
                float(invasion_count > len(interaction_arr) * 0.3)  # Frequent invasion
            ])
        else:
            features.extend([0] * 20)
    else:
        features.extend([0] * 20)
    
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
