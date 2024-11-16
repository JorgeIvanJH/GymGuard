import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolo11n-pose.pt')

# Run inference on the source
results = model.track(source=0, show=True, tracker="bytetrack.yaml")
