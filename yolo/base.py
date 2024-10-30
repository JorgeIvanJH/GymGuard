import os
import cv2
from ultralytics import solutions

# Set the environment variable to avoid OpenMP runtime error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

cap = cv2.VideoCapture("videos/pushups1.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

gym = solutions.AIGym(
    model="yolo11n-pose.pt",
    show=True,
    kpts=[6, 8, 10, 5, 7, 9],
)

# Manually set lw if not set in the AIGym class
if not hasattr(gym, 'lw') or gym.lw is None:
    gym.lw = 1  # Set a default value for lw
    
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    im0 = gym.monitor(im0)

cv2.destroyAllWindows()