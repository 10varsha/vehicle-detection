import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import os


class Trainer:
    def __init__(self, dataset, val_dataset=None, num_classes=197, batch_size=4, lr=0.005, num_epochs=10, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

        # Prepare DataLoaders
        self.train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)
        if val_dataset:
            self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=self.collate_fn)
        else:
            self.val_loader = None

        # Load pretrained model
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        self.model.to(self.device)

        # Optimizer and LR scheduler
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=0.9, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            for images, targets in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                images = [image.to(self.device) for image in images]  # Move images to device
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]  # Move targets to device
                
                # Perform forward pass, loss calculation, and backpropagation
                loss_dict = self.model(images, targets)
                # Get total loss
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

            self.lr_scheduler.step()
            print(f"[INFO] Epoch {epoch+1} Loss: {epoch_loss:.4f}")

    def save_model(self, path="outputs/fine_tuned_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved to {path}")

    def get_model(self):
        return self.model
