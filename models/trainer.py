import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from models.vehicle_detector import VehicleDetector

class Trainer:
    def __init__(self, dataset, val_dataset=None, device=None, batch_size=4, lr=0.005, epochs=10):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs

        # DataLoader
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_fn) if val_dataset else None

        # Load model
        self.model = VehicleDetector(num_classes=197)  # 196 + 1 for background
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.SGD(self.model.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        # LR scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)