import os
import json
import cv2

from utils.visualizer import Visualizer
from utils.result_saver import ResultSaver
from models.vehicle_detection import VehicleDetector

def run_inference(image_folder, output_folder):
    # Initialize necessary objects
    detector = VehicleDetector(pretrained=True)  # Or load your fine-tuned model
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
                "boxes": boxes.tolist(),
                "labels": labels.tolist(),
                "scores": scores.tolist()
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

if __name__ == "__main__":
    # Paths for input images and output results
    image_folder = 'data/images/'  # Adjust according to your folder structure
    output_folder = 'outputs/'  # Adjust according to your desired output folder

    os.makedirs(os.path.join(output_folder, 'annotated_images'), exist_ok=True)  # Ensure the folder exists

    run_inference(image_folder, output_folder)



#python -m inference.run_inference
