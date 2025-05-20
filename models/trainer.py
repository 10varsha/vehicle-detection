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
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for images, targets in loop:
                # Move each image and target to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                # Backward and optimize
                losses.backward()
                self.optimizer.step()

                # Update progress bar
                loop.set_postfix(loss=losses.item())

            self.lr_scheduler.step()

    def save_model(self, path="outputs/fine_tuned_model.pth"):
        dir_name = os.path.dirname(path)
        if dir_name:  # Only make dirs if a directory path exists
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved to {path}")


    def get_model(self):
        return self.model
