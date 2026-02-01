"""
Contrastive Harassment Detection - Apple Silicon Optimized Pipeline
==================================================================
Optimized for:
- Apple Silicon M4 (12 CPU cores)
- 16GB RAM
- Large frame-based datasets
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool
from detect_pose import pose_model
from interaction_features import extract_sequence_features

# ------------------ CONFIG ------------------ #

FRAME_BASE = "data/frames/train"
SEQUENCE_LENGTH = 10
STRIDE = 5
NORMAL_STRIDE_MULT = 2

MAX_WORKERS = 8   # optimal for M4 + 16GB

HARASSMENT_CATEGORIES = {
    'Abuse': {'label': 1, 'max_videos': 25},
    'Assault': {'label': 1, 'max_videos': 25},
    'Fighting': {'label': 1, 'max_videos': 20},
    'NormalVideos': {'label': 0, 'max_videos': 50}
}

SECONDARY_CATEGORIES = {
    'Arrest': 1, 'Arson': 1, 'Burglary': 1,
    'Robbery': 1, 'Shooting': 1, 'Vandalism': 1
}

# ------------------ CORE FUNCTIONS ------------------ #

def process_single_video(args):
    """
    Fully processes ONE video folder:
    - Loads frames
    - Runs pose ONCE per frame
    - Extracts sliding-window features
    """
    video_path, label, stride = args

    frame_files = sorted(
        f for f in os.listdir(video_path)
        if f.endswith(('.jpg', '.png'))
    )

    if len(frame_files) < SEQUENCE_LENGTH:
        return []

    # ---- POSE CACHE (major speedup) ---- #
    pose_cache = {}
    for f in frame_files:
        img = cv2.imread(os.path.join(video_path, f))
        if img is None:
            continue
        pose_cache[f] = pose_model(img, verbose=False)

    X_local = []

    for start in range(0, len(frame_files) - SEQUENCE_LENGTH + 1, stride):
        seq_frames = frame_files[start:start + SEQUENCE_LENGTH]
        person_tracks = defaultdict(list)

        for f in seq_frames:
            results = pose_cache.get(f)
            if not results:
                continue

            for pid, kp in enumerate(results[0].keypoints):
                if pid < 2:
                    person_tracks[pid].append(
                        kp.xy.cpu().numpy()[0]
                    )

        if not person_tracks:
            continue

        min_len = min(len(v) for v in person_tracks.values())
        if min_len < SEQUENCE_LENGTH // 2:
            continue

        for pid in person_tracks:
            person_tracks[pid] = person_tracks[pid][:min_len]

        features = extract_sequence_features(
            person_tracks, SEQUENCE_LENGTH
        )
        X_local.append((features, label))

    return X_local


def process_category(category, cfg):
    print(f"\n🔹 Processing {category}")

    base_path = os.path.join(FRAME_BASE, category)
    if not os.path.isdir(base_path):
        return [], []

    videos = sorted(os.listdir(base_path))[:cfg['max_videos']]

    stride = STRIDE * NORMAL_STRIDE_MULT if category == "NormalVideos" else STRIDE

    tasks = [
        (os.path.join(base_path, v), cfg['label'], stride)
        for v in videos
        if os.path.isdir(os.path.join(base_path, v))
    ]

    X_cat, y_cat = [], []

    with Pool(processes=MAX_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(process_single_video, tasks),
            total=len(tasks),
            desc=f"{category}"
        ):
            for feat, lbl in result:
                X_cat.append(feat)
                y_cat.append(lbl)

    return X_cat, y_cat

# ------------------ MAIN ------------------ #

def main():
    X, y = [], []

    for cat, cfg in HARASSMENT_CATEGORIES.items():
        Xc, yc = process_category(cat, cfg)
        X.extend(Xc)
        y.extend(yc)

        # Save incrementally (fault-tolerant)
        np.save(f"data/features/X_{cat}.npy", np.array(Xc))
        np.save(f"data/features/y_{cat}.npy", np.array(yc))

    for cat, label in SECONDARY_CATEGORIES.items():
        Xc, yc = process_category(cat, {'label': label, 'max_videos': 10})
        X.extend(Xc)
        y.extend(yc)

    X = np.array(X)
    y = np.array(y)

    os.makedirs("data/features", exist_ok=True)
    np.save("data/features/X_harassment_sequence.npy", X)
    np.save("data/features/y_harassment_sequence.npy", y)

    print("\n✅ DONE")
    print(f"Total sequences: {len(X):,}")
    print(f"Normal: {np.sum(y==0)} | Abnormal: {np.sum(y==1)}")

if __name__ == "__main__":
    main()