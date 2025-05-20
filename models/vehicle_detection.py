from models.detector import Detector
import torchvision
import torch

# Define device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

class VehicleDetector(Detector):
    def __init__(self, device=None):
        super().__init__('fasterrcnn_resnet50_fpn', device)
        print("[INFO] Using COCO pretrained model for car detection.")

    def predict(self, image_path, threshold=0.5):
        boxes, labels, scores = super().predict(image_path, threshold)
        # Filter for 'car' (COCO class ID 3)
        filtered = [(b, l, s) for b, l, s in zip(boxes, labels, scores) if l == 3 and s >= threshold]
        if filtered:
            boxes, labels, scores = zip(*filtered)
        else:
            boxes, labels, scores = [], [], []
        return boxes, labels, scores
