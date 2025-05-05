import os
from inference.run_inference import run_inference
from models.trainer import Trainer
from data.dataset import StanfordCarsDataset
import torchvision.transforms.v2 as T
import torch

if __name__ == "__main__":
    train_transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True)
    ])

    dataset = StanfordCarsDataset(
        root_dir="data/raw/cars_train",
        annotation_file="data/raw/devkit/cars_train_annos.mat",
        transforms=train_transforms
    )

    trainer = Trainer(dataset=dataset, num_epochs=10)

    model_path = "outputs/fine_tuned_model.pth"

    if os.path.exists(model_path):
        print("[INFO] Loading pretrained weights...")
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
    else:
        trainer.train()
        trainer.save_model(model_path)

    # Run inference
    image_folder = "data/raw/cars_test"
    output_folder = "outputs/inference"
    run_inference(image_folder, output_folder, model_path=model_path)
