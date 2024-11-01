filename_ext = "my-squats.jpg"

filename, file_extension = filename_ext.split('.')
reading_path = "videos/input/"
saving_path = "videos/output/"

from ultralytics import YOLO
import pandas as pd
import cv2
model = YOLO("yolo11n-pose.pt")

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

cols = ["frame", "person"]
cols = cols + list(keypoint_map.values())
df = pd.DataFrame(columns=cols)

results = model(source = reading_path + filename_ext, show = True, save = False, conf = 0.3)
# print("results: ", results, dir(results))
for frame_idx, frame_result in enumerate(results):
    cv2.imwrite(saving_path + filename_ext, frame_result.orig_img)
    # print(f"Frame {frame_idx}: {frame_result}")
    # print("frame_result: ", type(frame_result), frame_result, dir(frame_result))
    df.loc[frame_idx,"frame"] = frame_idx
    for person_idx, person in enumerate(frame_result.keypoints):
        # print(f"\tPerson {person_idx}: {person}")
        # print("\tperson: ", type(person), person, dir(person))
        df.loc[frame_idx,"person"] = person_idx
        for joint_idx, joint in enumerate(person):
            # print(f"\t\tJoint {joint_idx}: {joint}")
            # print("\t\tjoint: ", type(joint), joint, dir(joint))
            joints_xy = joint.xy
            df.loc[frame_idx,list(keypoint_map.values())] = joints_xy[0,:,:].tolist()
            
df.to_csv(saving_path + filename + ".csv", index = False)
