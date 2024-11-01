import os
import cv2
from ultralytics import solutions
# from ultralytics.ultralytics import solutions
filename = "my_squats.mp4"
reading_path = "videos/input/"
saving_path = "videos/output/"

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("reading_path + filename: ", reading_path + filename)

cap = cv2.VideoCapture(reading_path + filename)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter( saving_path + filename[:-4] + ".avi" , cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


model = solutions.AIGym(
    model="yolo11n-pose.pt",
    show=True,
    lw=1,  # Line width
    kpts=[16,14,12]
)
print("model: ", type(model), model, dir(model))

if not hasattr(model, 'lw') or model.lw is None:
    model.lw = 1  # Set a default value for lw
    
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    print("model.kpts: ", type(model.kpts), model.kpts, dir(model.kpts))
    im0 = model.monitor(im0)
    video_writer.write(im0)

    print("im0: ", type(im0), im0, dir(im0))
    1/0

cv2.destroyAllWindows()
video_writer.release()






#################### APPROACH 2 ####################

import cv2
from ultralytics import YOLO
import csv
import time

filename = "my_squats.mp4"
reading_path = "videos/input/"
saving_path = "videos/output/"

# Load a model
model = YOLO("yolov8n-pose.pt")  # Replace with a higher model variant if needed

cap = cv2.VideoCapture(reading_path + filename)

joint_positions_over_time = []  # List to store joint data for each frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO pose detection on the frame
    results = model(frame)
    print("results: ", results,dir(results))
    # Capture the timestamp for each frame (optional)
    timestamp = time.time()

    # Loop over detected poses (each person detected in the frame)
    for r in results:
        frame_joints = {'timestamp': timestamp}  # Initialize dictionary for this frame
        keypoints = r.keypoints.xy.cpu().numpy()  # Extract keypoints for this detection
        print("r: ", type(r), r, dir(r))
        print("r.keypoints: ", type(r.keypoints), r.keypoints, dir(r.keypoints))
        print("r.keypoints.xy: ", type(r.keypoints.xy),r.keypoints.xy, dir(r.keypoints.xy))
        confidences = r.keypoints.conf.cpu().numpy() if r.keypoints.has_visible else None

        # Parse and save each keypoint position
        for idx, (x, y) in enumerate(keypoints):
            frame_joints[f'joint_{idx}_x'] = x
            frame_joints[f'joint_{idx}_y'] = y
            if confidences is not None:
                frame_joints[f'joint_{idx}_confidence'] = confidences[idx]  # Confidence score for the keypoint

        joint_positions_over_time.append(frame_joints)

cap.release()

# Optionally, save the joint history to a file
with open('joint_history.txt', 'w') as f:
    for frame_joints in joint_positions_over_time:
        f.write(f"{frame_joints}\n")