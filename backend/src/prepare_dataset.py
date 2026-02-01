import numpy as np
import os
import cv2
from tqdm import tqdm
from config import LABEL_MAP
from feature_extract import extract_features_from_frame
from detect_pose import pose_model

X, y = [], []

FRAME_BASE = "data/frames/train"

print("Preparing dataset from extracted frames...")

for category in tqdm(os.listdir(FRAME_BASE), desc="Processing categories"):
    category_path = os.path.join(FRAME_BASE, category)
    
    if not os.path.isdir(category_path):
        continue
    
    label = LABEL_MAP.get(category, 0)
    
    for video_folder in tqdm(os.listdir(category_path), desc=f"Processing {category}", leave=False):
        video_path = os.path.join(category_path, video_folder)
        
        if not os.path.isdir(video_path):
            continue
        
        # Extract sequence features for this video
        video_features = []
        
        frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        
        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            features = extract_features_from_frame(frame_path, pose_model)
            video_features.append(features)
        
 
        if len(video_features) == 0:
            continue
        
        aggregated = np.mean(video_features, axis=0)
        
        X.append(aggregated)
        y.append(label)


os.makedirs("data/features", exist_ok=True)
np.save("data/features/X.npy", np.array(X))
np.save("data/features/y.npy", np.array(y))

print(f"\nDataset preparation complete!")
print(f"Total samples: {len(X)}")
print(f"Feature shape: {np.array(X).shape}")
print(f"Label distribution: Normal={sum(np.array(y)==0)}, Abnormal={sum(np.array(y)==1)}")