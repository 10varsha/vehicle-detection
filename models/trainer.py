import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
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

        # If no val dataset is provided, split from training
        if not val_dataset:
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            self.dataset, self.val_dataset = random_split(dataset, [train_size, val_size]) #

        # Prepare DataLoaders
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=self.collate_fn)

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

    def sanitize_targets(self, targets, width, height):
        clean_targets = []
        for t in targets:
            boxes = t['boxes'].clone()
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            clean_targets.append({
                'boxes': boxes[keep],
                'labels': t['labels'][keep]
            })
        return clean_targets

    def train(self):
        self.model.train()

        for epoch in range(self.num_epochs):
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for images, targets in loop:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Optional: clamp boxes
                width, height = images[0].shape[2], images[0].shape[1]
                targets = self.sanitize_targets(targets, width, height)

                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                if not torch.isnan(losses):
                    losses.backward()
                    self.optimizer.step()
                    loop.set_postfix(loss=losses.item())
                else:
                    loop.set_postfix(loss="NaN - skipped")

            self.lr_scheduler.step()

    def evaluate(self, iou_threshold=0.5):
        self.model.eval()
        total = 0
        correct = 0

        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc="Evaluating"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                outputs = self.model(images)

                for output, target in zip(outputs, targets):
                    pred_boxes = output['boxes']
                    true_boxes = target['boxes']

                    if len(pred_boxes) == 0 or len(true_boxes) == 0:
                        continue

                    ious = box_iou(pred_boxes, true_boxes)
                    max_ious, _ = ious.max(dim=1)

                    matches = (max_ious > iou_threshold).sum().item()
                    correct += matches
                    total += len(true_boxes)

        accuracy = correct / total if total > 0 else 0
        print(f"[RESULTS] IoU@{iou_threshold:.2f} Accuracy: {accuracy:.4f}")

    def test_single_image(self, image_path, confidence_threshold=0.5):
        self.model.eval()
        image = Image.open(image_path).convert("RGB")
        image_tensor = F.to_tensor(image).to(self.device)

        with torch.no_grad():
            output = self.model([image_tensor])[0]

        boxes = output['boxes'][output['scores'] > confidence_threshold]
        labels = output['labels'][output['scores'] > confidence_threshold]

        print(f"[INFO] Detected {len(boxes)} object(s)")

        image_drawn = draw_bounding_boxes((image_tensor * 255).byte().cpu(), boxes.cpu(),
                                          labels=[str(l.item()) for l in labels],
                                          colors="green", width=2)
        plt.imshow(to_pil_image(image_drawn))
        plt.axis("off")
        plt.show()

    def get_model(self):
        return self.model

    def save_model(self, path="outputs/fine_tuned_model.pth"):
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"[INFO] Model saved to {path}")
