import os
import sys
sys.path.append('..')
from cam import extract_frames

BASE_DIR = "data/Train"
OUT_DIR = "data/frames/train"

for category in os.listdir(BASE_DIR):
    category_path = os.path.join(BASE_DIR, category)

    for video in os.listdir(category_path):
        video_path = os.path.join(category_path, video)

        out_folder = os.path.join(
            OUT_DIR,
            category,
            video.split(".")[0]
        )
        os.makedirs(out_folder, exist_ok=True)

        extract_frames(video_path, out_folder)