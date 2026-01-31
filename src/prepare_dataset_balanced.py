import numpy as np
import os
import cv2
from tqdm import tqdm
from config import LABEL_MAP
from feature_extract import extract_features_from_frame
from detect_pose import pose_model

X, y = [], []

FRAME_BASE = "data/frames/train"

# BALANCED MODE: Process more samples with better distribution
MAX_VIDEOS_PER_CATEGORY = 20  # More videos per category
MAX_FRAMES_PER_VIDEO = 15     # More frames per video

print("🚀 BALANCED MODE: Processing data for better accuracy...")
print(f"   - {MAX_VIDEOS_PER_CATEGORY} videos per category")
print(f"   - {MAX_FRAMES_PER_VIDEO} frames per video\n")

for category in tqdm(os.listdir(FRAME_BASE), desc="Processing categories"):
    category_path = os.path.join(FRAME_BASE, category)
    
    if not os.path.isdir(category_path):
        continue
    
    label = LABEL_MAP.get(category, 0)
    
    video_count = 0
    for video_folder in os.listdir(category_path):
        if video_count >= MAX_VIDEOS_PER_CATEGORY:
            break
            
        video_path = os.path.join(category_path, video_folder)
        
        if not os.path.isdir(video_path):
            continue
        
        video_features = []
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))])
        frame_files = frame_files[:MAX_FRAMES_PER_VIDEO]
        
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            features = extract_features_from_frame(frame_path, pose_model)
            video_features.append(features)
        
        if len(video_features) == 0:
            continue
        
        aggregated = np.mean(video_features, axis=0)
        X.append(aggregated)
        y.append(label)
        
        # Data augmentation: add max aggregation too
        max_agg = np.max(video_features, axis=0)
        X.append(max_agg)
        y.append(label)
        
        video_count += 1

# Save features
os.makedirs("data/features", exist_ok=True)
np.save("data/features/X.npy", np.array(X))
np.save("data/features/y.npy", np.array(y))

print(f"\n✅ Balanced dataset preparation complete!")
print(f"   Total samples: {len(X)}")
print(f"   Feature shape: {np.array(X).shape}")
print(f"   Normal: {sum(np.array(y)==0)}, Abnormal: {sum(np.array(y)==1)}")
