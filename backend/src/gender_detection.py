"""
Gender Detection Module for CivicGuard
Uses body pose characteristics for gender estimation
This is used ONLY as a contextual signal, not as a standalone decision factor.
"""
import numpy as np
import cv2


def estimate_gender_from_pose(keypoints):
    """
    Estimate gender from pose keypoints using body structure analysis.
    
    This is a simplified heuristic-based approach that uses:
    - Shoulder-to-hip ratio
    - Body height-to-width ratio
    - Torso aspect ratio
    
    Args:
        keypoints: YOLO pose keypoints (17x3 array with x, y, confidence)
    
    Returns:
        'male', 'female', or 'unknown'
        confidence score (0-1)
    
    Note: This is an APPROXIMATION and should ONLY be used as contextual information
    """
    if keypoints is None or len(keypoints) < 13:
        return 'unknown', 0.0
    
    kp = np.array(keypoints)
    
    # Handle both (17, 2) and (17, 3) keypoint formats
    # (17, 2) = x, y only (from existing datasets)
    # (17, 3) = x, y, confidence (from YOLO pose detection)
    has_confidence = kp.shape[1] >= 3 if len(kp.shape) > 1 else False
    
    # Helper function to check if keypoint is valid
    def is_valid_kp(idx):
        if has_confidence:
            return kp[idx][2] > 0.5  # Check confidence
        else:
            return kp[idx][0] > 0  # Check if x coordinate exists
    
    # Extract key body points
    left_shoulder = kp[5][:2] if is_valid_kp(5) else None
    right_shoulder = kp[6][:2] if is_valid_kp(6) else None
    left_hip = kp[11][:2] if is_valid_kp(11) else None
    right_hip = kp[12][:2] if is_valid_kp(12) else None
    
    # Need at least shoulders and hips
    if any(x is None for x in [left_shoulder, right_shoulder, left_hip, right_hip]):
        return 'unknown', 0.0
    
    # Calculate shoulder width
    shoulder_width = np.linalg.norm(right_shoulder - left_shoulder)
    
    # Calculate hip width
    hip_width = np.linalg.norm(right_hip - left_hip)
    
    # Shoulder-to-hip ratio (males typically have broader shoulders relative to hips)
    if hip_width > 0:
        shoulder_hip_ratio = shoulder_width / hip_width
    else:
        return 'unknown', 0.0
    
    # Calculate torso height
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_height = np.linalg.norm(shoulder_center - hip_center)
    
    # Height-to-shoulder-width ratio
    if shoulder_width > 0:
        height_width_ratio = torso_height / shoulder_width
    else:
        return 'unknown', 0.0
    
    # Heuristic classification
    # Males tend to have: ratio > 1.0 (broader shoulders)
    # Females tend to have: ratio < 0.95 (broader hips or equal)
    
    confidence = 0.5
    gender = 'unknown'
    
    # Conservative thresholds to reduce misclassification
    if shoulder_hip_ratio > 1.05:
        gender = 'male'
        confidence = min(0.7, 0.5 + (shoulder_hip_ratio - 1.05) / 2)
    elif shoulder_hip_ratio < 0.92:
        gender = 'female'
        confidence = min(0.7, 0.5 + (0.92 - shoulder_hip_ratio) / 2)
    else:
        # Ambiguous range
        gender = 'unknown'
        confidence = 0.3
    
    return gender, float(confidence)


def detect_gender_interaction_pattern(person1_kp, person2_kp, person1_gender, person2_gender, 
                                       person1_center, person2_center, frame_history=None):
    """
    Detect gender-based interaction patterns that may indicate harassment.
    
    This function checks for:
    - Male approaching/following female
    - Sustained proximity with gender asymmetry
    - Movement alignment (following behavior)
    
    Args:
        person1_kp: Keypoints of person 1
        person2_kp: Keypoints of person 2
        person1_gender: Estimated gender of person 1 ('male', 'female', 'unknown')
        person2_gender: Estimated gender of person 2
        person1_center: Center position of person 1
        person2_center: Center position of person 2
        frame_history: Optional history of positions for tracking (list of dicts)
    
    Returns:
        Dictionary with gender-aware interaction features
    """
    features = {
        'male_to_female_proximity': 0.0,
        'sustained_approach': 0.0,
        'movement_alignment': 0.0,
        'following_score': 0.0,
        'avoidance_detected': 0.0,
        'gender_interaction_risk': 0.0
    }
    
    # Only compute if we have gender information
    if person1_gender == 'unknown' or person2_gender == 'unknown':
        return features
    
    # Check if this is a male-female interaction
    male_to_female = (person1_gender == 'male' and person2_gender == 'female')
    female_to_male = (person1_gender == 'female' and person2_gender == 'male')
    
    if not (male_to_female or female_to_male):
        return features
    
    # Determine who is the potential aggressor (male) and target (female)
    if male_to_female:
        aggressor_center = person1_center
        target_center = person2_center
        aggressor_kp = person1_kp
        target_kp = person2_kp
    else:
        aggressor_center = person2_center
        target_center = person1_center
        aggressor_kp = person2_kp
        target_kp = person1_kp
    
    # Calculate interpersonal distance
    distance = np.linalg.norm(aggressor_center - target_center)
    
    # Proximity score (higher when very close)
    if distance < 50:
        proximity_score = 1.0
    elif distance < 100:
        proximity_score = 0.7
    elif distance < 150:
        proximity_score = 0.4
    else:
        proximity_score = 0.1
    
    features['male_to_female_proximity'] = proximity_score
    
    # Analyze movement patterns from history
    if frame_history and len(frame_history) > 3:
        # Extract position history
        aggressor_positions = [h.get('aggressor_pos') for h in frame_history if 'aggressor_pos' in h]
        target_positions = [h.get('target_pos') for h in frame_history if 'target_pos' in h]
        
        if len(aggressor_positions) > 3 and len(target_positions) > 3:
            # Check if distance is decreasing (sustained approach)
            distances = [np.linalg.norm(np.array(a) - np.array(t)) 
                        for a, t in zip(aggressor_positions, target_positions)]
            
            if len(distances) > 1:
                distance_trend = np.mean(np.diff(distances))
                if distance_trend < 0:  # Getting closer
                    features['sustained_approach'] = min(1.0, abs(distance_trend) / 10)
            
            # Check movement alignment (following)
            aggressor_movements = np.diff(aggressor_positions, axis=0)
            target_movements = np.diff(target_positions, axis=0)
            
            if len(aggressor_movements) > 0 and len(target_movements) > 0:
                # Normalize movements
                aggressor_dirs = aggressor_movements / (np.linalg.norm(aggressor_movements, axis=1, keepdims=True) + 1e-6)
                target_dirs = target_movements / (np.linalg.norm(target_movements, axis=1, keepdims=True) + 1e-6)
                
                # Calculate alignment
                alignments = np.sum(aggressor_dirs * target_dirs, axis=1)
                features['movement_alignment'] = np.mean(alignments[alignments > 0])
                
                # Following score combines alignment and approach
                features['following_score'] = (features['movement_alignment'] + features['sustained_approach']) / 2
            
            # Detect avoidance (target moving away consistently)
            if len(target_movements) > 2:
                # Vector from target to aggressor
                escape_vectors = [np.array(t) - np.array(a) 
                                for a, t in zip(aggressor_positions[-3:], target_positions[-3:])]
                target_move_vectors = target_movements[-2:]
                
                # Check if target is moving in same direction as escape vector (away from aggressor)
                if len(escape_vectors) > 0 and len(target_move_vectors) > 0:
                    escape_dirs = escape_vectors / (np.linalg.norm(escape_vectors, axis=1, keepdims=True) + 1e-6)
                    move_dirs = target_move_vectors / (np.linalg.norm(target_move_vectors, axis=1, keepdims=True) + 1e-6)
                    
                    avoidance_alignments = np.sum(escape_dirs[:-1] * move_dirs, axis=1)
                    features['avoidance_detected'] = np.mean(avoidance_alignments[avoidance_alignments > 0.5])
    
    # Calculate overall gender interaction risk
    # This combines proximity, approach, following, and avoidance
    risk_components = [
        features['male_to_female_proximity'] * 0.3,
        features['sustained_approach'] * 0.25,
        features['following_score'] * 0.25,
        features['avoidance_detected'] * 0.20
    ]
    
    features['gender_interaction_risk'] = sum(risk_components)
    
    return features


def should_apply_gender_aware_scoring(person1_gender, person2_gender, 
                                       person1_confidence, person2_confidence,
                                       min_confidence=0.5):
    """
    Determine if gender-aware scoring should be applied.
    
    Gender detection should only influence scoring when:
    1. Both genders are detected with sufficient confidence
    2. There is a male-female interaction
    
    Args:
        person1_gender: Gender of person 1
        person2_gender: Gender of person 2
        person1_confidence: Confidence of person 1 gender detection
        person2_confidence: Confidence of person 2 gender detection
        min_confidence: Minimum confidence threshold
    
    Returns:
        bool: Whether to apply gender-aware scoring
    """
    # Must have sufficient confidence
    if person1_confidence < min_confidence or person2_confidence < min_confidence:
        return False
    
    # Must be male-female interaction
    if (person1_gender == 'male' and person2_gender == 'female') or \
       (person1_gender == 'female' and person2_gender == 'male'):
        return True
    
    return False
