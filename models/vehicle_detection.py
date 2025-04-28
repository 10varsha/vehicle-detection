from models.detector import Detector
import torchvision

class VehicleDetector(Detector):
    def __init__(self, num_classes=196+1, model_name='fasterrcnn_resnet50_fpn', device=None, pretrained=True):
        #num_classes: 196 car types + 1 background
        super().__init__(model_name, device)

        # Replace the classification head (this is MANDATORY)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        print(f"[INFO] VehicleDetector initialized with {num_classes} classes.")

    def predict(self, image_path, threshold=0.5):
        # Use parent Detector's preprocessing + inference
        boxes, labels, scores = super().predict(image_path, threshold)
        return boxes, labels, scores