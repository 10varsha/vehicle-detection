import os
import json
import cv2
import numpy as np


from utils.visualizer import Visualizer
from utils.result_saver import ResultSaver
from models.vehicle_detection import VehicleDetector

def run_inference(image_folder, output_folder):
    # Initialize necessary objects
    detector = VehicleDetector()  # Or load your fine-tuned model
    visualizer = Visualizer()
    result_saver = ResultSaver()

    # Results storage
    all_results = []

    # Loop through all images in the folder
    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)

        if image_filename.endswith(('.jpg', '.png', '.jpeg')):
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Run inference
            boxes, labels, scores = detector.predict(image_path)

            # Save results to JSON
            image_result = {
                "filename": image_filename,
                "boxes": np.array(boxes).tolist(),
                "labels": np.array(labels).tolist(),
                "scores": np.array(scores).tolist()
            }

            all_results.append(image_result)

            # Annotate image with bounding boxes and labels
            annotated_image = visualizer.draw_boxes(image.copy(), boxes, labels, scores)

            # Save annotated image
            annotated_image_path = os.path.join(output_folder, "annotated_images", f"{image_filename}")
            cv2.imwrite(annotated_image_path, annotated_image)

    # Save all results in JSON
    results_json_path = os.path.join(output_folder, "results.json")
    with open(results_json_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

    print(f"Inference completed! Results saved in {output_folder}.")

COCO_INSTANCE_CATEGORY_NAMES = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    # Add more if needed...
}

if __name__ == "__main__":
    # Paths for input images and output results
    image_folder = 'data/raw/cars_test'  # Adjust according to your folder structure
    output_folder = 'outputs/inference'  # Adjust according to your desired output folder

    os.makedirs(os.path.join(output_folder, 'annotated_images'), exist_ok=True)  # Ensure the folder exists

    run_inference(image_folder, output_folder, model_path = None)



#python -m inference.run_inference
