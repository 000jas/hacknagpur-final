from ultralytics import YOLO
import cv2

pose_model = YOLO("yolo/yolov8n-pose.pt")

def detect_pose(frame):
    results = pose_model(frame)
    persons = []
    for kp in results[0].keypoints.xy:
        persons.append(kp.numpy())
    return persons