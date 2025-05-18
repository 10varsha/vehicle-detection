import os
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
import scipy.io
import torch

class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotations = self.load_annotations(annotation_file)

    def load_annotations(self, annotation_file):
        mat = scipy.io.loadmat(annotation_file)
        raw_annos = mat['annotations'][0]
        cleaned = []

        for anno in raw_annos:
            try:
                # Extract filename first, safely
                filename = str(anno[5][0]) if len(anno) == 6 else str(anno[4][0])

                # Then extract other fields
                xmin = int(anno[0][0])
                ymin = int(anno[1][0])
                xmax = int(anno[2][0])
                ymax = int(anno[3][0])

                # Handle optional class_id
                if len(anno) == 6:
                    class_id = int(anno[4][0])
                else:
                    class_id = -1  # No label, test set

                cleaned.append({
                    'filename': filename,
                    'bbox': [xmin, ymin, xmax, ymax],
                    'label': class_id
                })

            except Exception as e:
                print(f"[WARN] Failed to parse annotation: {anno} | Error: {e}")
        
        print(f"[INFO] Loaded {len(cleaned)} cleaned annotations.")
        return cleaned


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        image_path = os.path.join(self.root_dir, item['filename'])
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[ERROR] Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        target = {
            'boxes': torch.tensor([item['bbox']], dtype=torch.float32),
            'labels': torch.tensor([item['label']], dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target
