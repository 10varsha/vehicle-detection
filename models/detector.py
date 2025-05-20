import torch
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image

# Define device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

class Detector:
    def __init__(self, model_name='fasterrcnn_resnet50_fpn', device=None):
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model()
        self.model.to(device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor()
        ])

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


    def predict(self, image_path, threshold=0.3):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            device = next(self.model.parameters()).device
            image_tensor = image_tensor.to(device)
            outputs = self.model(image_tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        # Filter based on threshold
        filtered = scores >= threshold
        return boxes[filtered], labels[filtered], scores[filtered]

