import numpy as np
import cv2
from gender_detection import estimate_gender_from_pose, detect_gender_interaction_pattern, should_apply_gender_aware_scoring


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


def extract_gender_aware_interaction_features(person1_kp, person2_kp, position_history=None):
    """
    Extract interaction features WITH gender-aware harassment indicators.
    
    This enhances standard interaction features with gender-based context:
    - Detects male-to-female proximity patterns
    - Identifies sustained approach and following behavior
    - Recognizes avoidance cues from potential target
    
    Gender is used ONLY as contextual signal, not standalone decision factor.
    
    Args:
        person1_kp: Keypoints of first person (17x3 array with confidence)
        person2_kp: Keypoints of second person
        position_history: Optional list of previous positions for temporal analysis
    
    Returns:
        Extended interaction features (15 standard + 6 gender-aware = 21 total)
    """
    # Get standard interaction features (15 features)
    standard_features = extract_interaction_features(person1_kp, person2_kp)
    
    if person1_kp is None or person2_kp is None:
        return standard_features + [0.0] * 6  # Add 6 zero gender features
    
    # Estimate genders from pose
    p1_gender, p1_gender_conf = estimate_gender_from_pose(person1_kp)
    p2_gender, p2_gender_conf = estimate_gender_from_pose(person2_kp)
    
    # Calculate centers for position tracking
    p1 = np.array(person1_kp)
    p2 = np.array(person2_kp)
    p1_center = np.mean(p1[p1[:, 0] > 0, :2], axis=0) if np.any(p1[:, 0] > 0) else np.array([0, 0])
    p2_center = np.mean(p2[p2[:, 0] > 0, :2], axis=0) if np.any(p2[:, 0] > 0) else np.array([0, 0])
    
    # Initialize gender-aware features
    gender_features = [0.0] * 6
    
    # Only apply gender-aware scoring if we have sufficient confidence
    if should_apply_gender_aware_scoring(p1_gender, p2_gender, p1_gender_conf, p2_gender_conf):
        # Get gender-specific interaction patterns
        gender_patterns = detect_gender_interaction_pattern(
            person1_kp, person2_kp, p1_gender, p2_gender,
            p1_center, p2_center, position_history
        )
        
        # Extract the 6 gender-aware features
        gender_features = [
            gender_patterns['male_to_female_proximity'],
            gender_patterns['sustained_approach'],
            gender_patterns['movement_alignment'],
            gender_patterns['following_score'],
            gender_patterns['avoidance_detected'],
            gender_patterns['gender_interaction_risk']
        ]
    
    # Combine standard features + gender-aware features
    return standard_features + gender_features



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
    
    CRITICAL: Harassment detection requires interaction context.
    - Single person scenarios return BASELINE MINIMAL RISK features
    - Risk scoring is conditioned on multi-person presence and sustained proximity
    
    Args:
        keypoint_sequences: Dict of {person_id: [kp_frame1, kp_frame2, ...]}
        sequence_length: Number of frames to analyze
    
    Returns:
        Feature vector for sequence-level classification
    """
    features = []
    
    person_ids = list(keypoint_sequences.keys())
    
    # ============================================================================
    # INTERACTION-DEPENDENT RISK LOGIC
    # ============================================================================
    # Harassment is inherently interaction-based and cannot occur in isolation.
    # When only one person is present, return baseline minimal risk features.
    # This prevents false positives from solo individuals moving normally.
    
    if len(person_ids) < 2:
        # FORCE MINIMAL BASELINE RISK for single-person scenarios
        # Return feature vector that will result in lowest possible risk score
        # Total: 40 temporal (2 people * 20) + 26 interaction/gender = 66 features
        baseline_features = [0.0] * 66
        
        # Set specific baseline indicators to signal "no threat"
        baseline_features[0] = 0.0   # No speed
        baseline_features[1] = 0.0   # No acceleration  
        baseline_features[40] = 999.0  # Maximum distance (no proximity) - first interaction feature
        baseline_features[60] = 0.0  # No following
        baseline_features[61] = 0.0  # No invasion
        
        return baseline_features
    # ============================================================================
    
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
        
        # ========================================================================
        # PROXIMITY AND INTERACTION VALIDATION
        # ========================================================================
        # Calculate average distance over sequence to determine if meaningful
        # interaction is occurring. Distant individuals with no proximity
        # should not generate elevated risk scores.
        
        distances = []
        for i in range(min(len(p1_seq), len(p2_seq))):
            if p1_seq[i] is not None and p2_seq[i] is not None:
                p1 = np.array(p1_seq[i])
                p2 = np.array(p2_seq[i])
                p1_center = np.mean(p1[p1[:, 0] > 0, :2], axis=0) if np.any(p1[:, 0] > 0) else None
                p2_center = np.mean(p2[p2[:, 0] > 0, :2], axis=0) if np.any(p2[:, 0] > 0) else None
                
                if p1_center is not None and p2_center is not None:
                    dist = np.linalg.norm(p1_center - p2_center)
                    distances.append(dist)
        
        # If people are consistently far apart (no meaningful interaction)
        # reduce risk by applying distance penalty
        avg_distance = np.mean(distances) if distances else 999.0
        proximity_factor = 1.0
        
        if avg_distance > 300:  # Very far - almost no interaction
            proximity_factor = 0.1
        elif avg_distance > 200:  # Moderate distance - reduce influence
            proximity_factor = 0.4
        elif avg_distance > 150:  # Social distance
            proximity_factor = 0.7
        # else: close proximity, full weight (proximity_factor = 1.0)
        
        # ========================================================================
        
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
            
            # Apply proximity factor to interaction-based features
            # This reduces risk when people are far apart
            mean_interaction = (mean_interaction * proximity_factor).tolist()
            features.extend(mean_interaction)
            
            # ====================================================================
            # GENDER-AWARE HARASSMENT INDICATORS
            # ====================================================================
            # Compute gender-aware features for enhanced harassment detection
            # This section adds 6 additional features when gender context is available
            
            # Build position history for temporal gender analysis
            position_history = []
            for i in range(len(p1_seq)):
                if i < len(p1_seq) and i < len(p2_seq):
                    if p1_seq[i] is not None and p2_seq[i] is not None:
                        p1 = np.array(p1_seq[i])
                        p2 = np.array(p2_seq[i])
                        p1_c = np.mean(p1[p1[:, 0] > 0, :2], axis=0) if np.any(p1[:, 0] > 0) else None
                        p2_c = np.mean(p2[p2[:, 0] > 0, :2], axis=0) if np.any(p2[:, 0] > 0) else None
                        
                        if p1_c is not None and p2_c is not None:
                            position_history.append({
                                'aggressor_pos': p1_c.tolist(),
                                'target_pos': p2_c.tolist()
                            })
            
            # Extract gender-aware features from most recent frame
            if len(p1_seq) > 0 and len(p2_seq) > 0:
                latest_p1 = p1_seq[-1]
                latest_p2 = p2_seq[-1]
                
                # Get gender-aware interaction features (includes standard 15 + gender 6)
                gender_aware_feats = extract_gender_aware_interaction_features(
                    latest_p1, latest_p2, position_history
                )
                
                # Extract only the 6 gender-specific features (last 6)
                gender_specific = gender_aware_feats[-6:]
                
                # Apply proximity factor to gender features as well
                gender_specific = [f * proximity_factor for f in gender_specific]
            else:
                gender_specific = [0.0] * 6
            
            # ====================================================================
            
            # Additional harassment indicators (with proximity factor applied)
            features.extend([
                distance_trend * proximity_factor,
                following_score * proximity_factor,
                float(invasion_count) * proximity_factor,
                float(following_score > 0.5 and distance_trend < 0) * proximity_factor,  # Persistent approach
                float(invasion_count > len(interaction_arr) * 0.3) * proximity_factor,  # Frequent invasion
                *gender_specific  # Add the 6 gender-aware features
            ])
        else:
            # No interaction data - pad with zeros (15 interaction + 5 harassment + 6 gender)
            features.extend([0] * 26)
    else:
        # No second person - pad with zeros
        features.extend([0] * 26)
    
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
