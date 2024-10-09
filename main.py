import torch
from ultralytics import YOLO

device: str = "mps" if torch.backends.mps.is_available() else "cpu"

# Create a new YOLO model from scratch
model = YOLO("yolov8n.yaml")
model.to(device)
# train the model
results = model.train(data = "confg.yaml", epochs = 225)
metrics = model.val()