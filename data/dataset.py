import os
import scipy.io
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


class StanfordCarsDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotations = scipy.io.loadmat(annotation_file)['annotations'][0]
        
        print(f"[INFO] Loaded {len(self.annotations)} annotations from {annotation_file}")

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        img_filename = str(anno['fname'][0])
        img_path = os.path.join(self.root, img_filename)
        image = Image.open(img_path).convert("RGB")

        # Bounding box coordinates
        xmin = int(anno['bbox_x1'][0][0])
        ymin = int(anno['bbox_y1'][0][0])
        xmax = int(anno['bbox_x2'][0][0])
        ymax = int(anno['bbox_y2'][0][0])
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        # Class label (1-indexed in dataset, keep it that way or subtract 1 if needed)
        label = int(anno['class'][0][0])
        labels = torch.tensor([label], dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (xmax - xmin) * (ymax - ymin)
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Wrap into torchvision tv_tensors for use with v2 transforms
        image = tv_tensors.Image(image)
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image))

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": torch.tensor([area], dtype=torch.float32),
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.annotations)
