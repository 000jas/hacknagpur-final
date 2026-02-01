import numpy as np
import os
import cv2
from tqdm import tqdm
from config import LABEL_MAP
from detect_pose import pose_model
from interaction_features import extract_sequence_features
from collections import defaultdict

X, y = [], []

FRAME_BASE = "data/frames/train"

# SEQUENCE-BASED MODE: Process temporal sequences for harassment detection
VIDEOS_PER_CATEGORY = 30
SEQUENCE_LENGTH = 10  # Number of consecutive frames to analyze
STRIDE = 5  # Overlap between sequences

print("🎯 SEQUENCE-BASED MODE: Processing temporal patterns for harassment detection...")
print(f"   - {VIDEOS_PER_CATEGORY} videos per category")
print(f"   - Sequence length: {SEQUENCE_LENGTH} frames")
print(f"   - Analyzing interaction patterns and temporal behavior\n")

for category in tqdm(os.listdir(FRAME_BASE), desc="Processing categories"):
    category_path = os.path.join(FRAME_BASE, category)
    
    if not os.path.isdir(category_path):
        continue
    
    label = LABEL_MAP.get(category, 0)
    
    video_count = 0
    for video_folder in os.listdir(category_path):
        if video_count >= VIDEOS_PER_CATEGORY:
            break
            
        video_path = os.path.join(category_path, video_folder)
        
        if not os.path.isdir(video_path):
            continue
        
        # Get all frames
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        
        if len(frame_files) < SEQUENCE_LENGTH:
            continue
        
        # Process sequences with sliding window
        for start_idx in range(0, len(frame_files) - SEQUENCE_LENGTH + 1, STRIDE):
            sequence_frames = frame_files[start_idx:start_idx + SEQUENCE_LENGTH]
            
            # Track keypoints for all detected persons across the sequence
            person_sequences = defaultdict(list)
            
            for frame_file in sequence_frames:
                frame_path = os.path.join(video_path, frame_file)
                frame = cv2.imread(frame_path)
                
                if frame is None:
                    continue
                
                # Detect all people in frame
                results = pose_model(frame, verbose=False)
                
                # Store keypoints for each person
                for person_id, keypoints_data in enumerate(results[0].keypoints):
                    if person_id < 2:  # Track up to 2 people for interaction
                        keypoints = keypoints_data.xy.cpu().numpy()[0]
                        person_sequences[person_id].append(keypoints)
            
            # Extract sequence-level features
            if len(person_sequences) > 0:
                # Ensure all sequences have same length
                min_length = min(len(seq) for seq in person_sequences.values())
                if min_length >= SEQUENCE_LENGTH // 2:  # At least half the sequence
                    # Truncate all sequences to same length
                    for pid in person_sequences:
                        person_sequences[pid] = person_sequences[pid][:min_length]
                    
                    features = extract_sequence_features(person_sequences, SEQUENCE_LENGTH)
                    
                    X.append(features)
                    y.append(label)
        
        video_count += 1

# Save features
os.makedirs("data/features", exist_ok=True)
np.save("data/features/X_sequence.npy", np.array(X))
np.save("data/features/y_sequence.npy", np.array(y))

print(f"\n✅ Sequence-based dataset preparation complete!")
print(f"   Total sequences: {len(X)}")
print(f"   Feature shape: {np.array(X).shape}")
print(f"   Normal: {sum(np.array(y)==0)}, Abnormal: {sum(np.array(y)==1)}")
print(f"\n📊 Feature breakdown:")
print(f"   - Temporal features: 40 (20 per person)")
print(f"   - Interaction features: 15 (interpersonal dynamics)")
print(f"   - Harassment indicators: 5 (following, invasion, etc.)")
