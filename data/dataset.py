import os
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import functional as F

class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms

        # Load .mat annotation file
        annos = scipy.io.loadmat(annotation_file)
        self.annotations = annos['annotations'][0]

        print(f"[INFO] Loaded {len(self.annotations)} annotations from {annotation_file}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        anno = self.annotations[idx]
        img_name = anno[5][0]
        label = int(anno[4][0][0]) - 1  # 1-based to 0-based

        # Load bounding box
        x1 = float(anno[0][0][0])
        y1 = float(anno[1][0][0])
        x2 = float(anno[2][0][0])
        y2 = float(anno[3][0][0])
        bbox = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

        # Load image
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = F.to_tensor(image)

        target = {
            "boxes": bbox,
            "labels": torch.tensor([label], dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target
