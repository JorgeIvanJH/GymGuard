
filename_ext = "my_squats_vid.mp4"

filename, file_extension = filename_ext.split('.')
reading_path = "videos/input/"
saving_path = "videos/output/"

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
from ultralytics import solutions
import pandas as pd
import time
import cv2

model = YOLO("yolo11n-pose.pt")
model_tracker = solutions.AIGym(
    model="yolo11n-pose.pt",
    show=True,
    lw=1,  # Line width
    kpts=[16,14,12]
)
if not hasattr(model_tracker, 'lw') or model_tracker.lw is None:
    model_tracker.lw = 1  # Set a default value for lw


keypoint_map = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

model.track(source = reading_path + filename_ext, show = True, save = True, conf = 0.3)

cap = cv2.VideoCapture(reading_path + filename_ext)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter( saving_path + filename + ".avi" , cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


cols = ["timestamp", "frame", "person"]
cols = cols + list(keypoint_map.values())
df = pd.DataFrame(columns=cols)
joint_positions_over_time = []  # List to store joint data for each frame

while cap.isOpened():

    success, img_frame = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    timestamp = time.time()
    # Run YOLO pose detection on the frame
    results = model(img_frame, show = True, save = False, conf = 0.3)
    # print("results: ", results, dir(results))
    for frame_idx, frame_result in enumerate(results):
        cv2.imwrite(saving_path + filename_ext, frame_result.orig_img)
        # print(f"Frame {frame_idx}: {frame_result}")
        # print("frame_result: ", type(frame_result), frame_result, dir(frame_result))
        df.loc[timestamp,"frame"] = frame_idx
        df.loc[timestamp,"timestamp"] = timestamp
        for person_idx, person in enumerate(frame_result.keypoints):
            # print(f"\tPerson {person_idx}: {person}")
            # print("\tperson: ", type(person), person, dir(person))
            df.loc[timestamp,"person"] = person_idx
            for joint_idx, joint in enumerate(person):
                # print(f"\t\tJoint {joint_idx}: {joint}")
                # print("\t\tjoint: ", type(joint), joint, dir(joint))
                joints_xy = joint.xy
                df.loc[timestamp,list(keypoint_map.values())] = joints_xy[0,:,:].tolist()

    im_monitor = model_tracker.monitor(img_frame)
    video_writer.write(im_monitor)
    
df.to_csv(saving_path + filename + ".csv", index = False)
cv2.destroyAllWindows()
video_writer.release()

# Optionally, save the joint history to a file
with open('joint_history.txt', 'w') as f:
    for frame_joints in joint_positions_over_time:
        f.write(f"{frame_joints}\n")