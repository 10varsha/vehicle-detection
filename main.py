import os
from inference.run_inference import run_inference
from models.trainer import Trainer
from models.test import Tester
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
        trainer.evaluate()
        trainer.save_model(model_path)

    # =================== TEST ON ANNOTATED TEST DATA ===================

    test_dataset = StanfordCarsDataset(
        root_dir="data/raw/cars_test",  # Make sure this exists
        annotation_file="data/raw/devkit/cars_test_annos.mat",  # Or whatever test anno file you have
        transforms=train_transforms
    )

    tester = Tester(model=trainer.get_model(), dataset=test_dataset)
    tester.test(num_samples=10)

    # =================== OPTIONAL INFERENCE ON RAW TEST IMAGES ===================

    image_folder = "data/raw/cars_test"
    output_folder = "outputs/inference"
    run_inference(image_folder, output_folder)

    