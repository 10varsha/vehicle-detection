import torch
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
from tqdm import tqdm

class Tester:
    def __init__(self, model, dataset, device=None, batch_size=1):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=self.collate_fn)

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def test(self, num_samples=10, iou_threshold=0.5):
        self.model.eval()
        correct = 0
        total = 0
        shown = 0

        with torch.no_grad():
            for images, targets in tqdm(self.loader, desc="Testing model"):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                outputs = self.model(images)

                for output, target in zip(outputs, targets):
                    shown += 1
                    print(f"\n[DEBUG] Sample #{shown}")

                    if not output or 'boxes' not in output or 'scores' not in output:
                        print("[WARNING] No valid prediction from model.")
                        continue

                    pred_boxes = output['boxes']
                    true_boxes = target['boxes']

                    print("Predicted Boxes:", pred_boxes)
                    print("True Boxes:", true_boxes)

                    if len(pred_boxes) == 0 or len(true_boxes) == 0:
                        print("[WARNING] Empty boxes detected.")
                        continue

                    ious = box_iou(pred_boxes, true_boxes)
                    max_ious, _ = ious.max(dim=1)
                    matches = (max_ious > iou_threshold).sum().item()

                    correct += matches
                    total += len(true_boxes)

                    if shown >= num_samples:
                        break


        acc = correct / total if total > 0 else 0
        print(f"[TEST RESULT] IoU@{iou_threshold:.2f} on {num_samples} samples: Accuracy = {acc:.4f}")
