from inference.run_inference import run_inference
from models.trainer import Trainer
from data.dataset import StanfordCarsDataset
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

if __name__ == "__main__":
    # 1. Set paths
    train_data_path = "data/raw/cars_train"
    annotation_file = "data/raw/devkit/cars_train_annos.mat"
    image_folder = "data/images/"
    output_folder = "outputs/"
    
    # 2. Create Dataset and DataLoader
    transforms = T.ToTensor()  # Replace with actual transforms pipeline
    dataset = StanfordCarsDataset(root=train_data_path, annotation_file=annotation_file, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # 3. Train the model
    trainer = Trainer(dataset=dataloader, num_epochs=10)
    trainer.train()
    trainer.save_model("outputs/fine_tuned_model.pth")

    # 4. Run inference using fine-tuned model
    run_inference(image_folder, output_folder, model_path="outputs/fine_tuned_model.pth")
