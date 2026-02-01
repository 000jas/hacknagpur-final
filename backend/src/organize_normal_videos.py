"""
Organize NormalVideos frames into video-specific directories
"""
import os
import shutil
from collections import defaultdict
from tqdm import tqdm

source_dir = "data/Train/NormalVideos"
target_dir = "data/frames/train/NormalVideos"

print("🗂️  Organizing NormalVideos frames by video...")

# Get all PNG files
frames = [f for f in os.listdir(source_dir) if f.endswith('.png')]

# Group by video name (everything before the frame number)
video_groups = defaultdict(list)
for frame in frames:
    # Extract video name (e.g., "Normal_Videos001_x264")
    parts = frame.rsplit('_', 1)[0]  # Remove frame number
    video_groups[parts].append(frame)

print(f"Found {len(video_groups)} unique videos")

# Create directories and move frames
for video_name, video_frames in tqdm(video_groups.items(), desc="Organizing"):
    video_dir = os.path.join(target_dir, video_name)
    os.makedirs(video_dir, exist_ok=True)
    
    for frame in sorted(video_frames):
        src = os.path.join(source_dir, frame)
        dst = os.path.join(video_dir, frame)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

print(f"✅ Organized {len(video_groups)} videos into {target_dir}")
