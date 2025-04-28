import torch
import torchvision
from torchvision import transforms
from PIL import Image

class Detector:
    def __init__(self, model_name='fasterrcnn_resnet50_fpn', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

    def load_model(self):
        # Loading a pretrained model from torchvision
        if self.model_name == 'fasterrcnn_resnet50_fpn':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        else:
            raise ValueError(f"Model {self.model_name} not supported yet. Add it in Detector class.")
        return model
    
    def preprocess(self, image_path):
        # Preprocess the input image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert image to tensor
        ])
        image = transform(image)
        return image
    
    def postprocess(self, outputs, threshold=0.5):
        # Post-process the model outputs to get bounding boxes above a certain score
        boxes = []
        labels = []
        scores = []

        for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
            if score >= threshold:
                boxes.append(box.cpu().detach().numpy())
                labels.append(label.cpu().detach().numpy())
                scores.append(score.cpu().detach().numpy())
        
        return boxes, labels, scores

