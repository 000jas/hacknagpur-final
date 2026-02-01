#!/usr/bin/env python3
"""
Organize existing frames into video folders for feature extraction
Since the UCF Crime dataset downloaded already contains frames, not videos
"""

import os
import shutil
from collections import defaultdict
from tqdm import tqdm

TRAIN_DIR = "data/Train"
FRAMES_DIR = "data/frames/train"

print("Organizing frames into video folders...")

for category in os.listdir(TRAIN_DIR):
    category_path = os.path.join(TRAIN_DIR, category)
    
    if not os.path.isdir(category_path):
        continue
    
    print(f"\nProcessing category: {category}")
    
    # Group frames by video name
    video_frames = defaultdict(list)
    
    for frame_file in tqdm(os.listdir(category_path), desc=f"Scanning {category}"):
        if not frame_file.endswith('.png'):
            continue
        
        # Extract video name (e.g., "Normal_Videos001_x264" from "Normal_Videos001_x264_0.png")
        parts = frame_file.rsplit('_', 1)
        if len(parts) == 2:
            video_name = parts[0]
            video_frames[video_name].append(frame_file)
    
    # Create folders and copy frames
    for video_name, frames in tqdm(video_frames.items(), desc=f"Organizing {category}"):
        video_folder = os.path.join(FRAMES_DIR, category, video_name)
        os.makedirs(video_folder, exist_ok=True)
        
        # Only process first 30 frames per video to speed up
        for frame_file in sorted(frames)[:30]:
            src = os.path.join(category_path, frame_file)
            dst = os.path.join(video_folder, frame_file)
            
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    print(f"✓ Organized {len(video_frames)} videos for {category}")

print("\n✅ Frame organization complete!")
