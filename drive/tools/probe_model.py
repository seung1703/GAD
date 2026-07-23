import torch
import sys

try:
    from ultralytics import YOLO
    model = YOLO('model/lane_seg_best.pt')
    print("It is a YOLO model.")
    sys.exit(0)
except Exception as e:
    print(f"Not YOLO: {e}")

try:
    ckpt = torch.load('model/lane_seg_best.pt', map_location='cpu')
    print("Keys in checkpoint:", ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))
except Exception as e:
    print(f"Error loading with torch: {e}")
