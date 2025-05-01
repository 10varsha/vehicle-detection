import json
import os
import cv2
import torch

# Define device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

class ResultSaver:
    def __init__(self, output_dir='outputs', annotated_dir='outputs/annotated_images'):
        self.output_dir = output_dir
        self.annotated_dir = annotated_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.annotated_dir, exist_ok=True)

        self.results = []
        print(f"[INFO] ResultSaver initialized. Saving to {self.output_dir} and {self.annotated_dir}")

    def add_result(self, image_name, boxes, labels, scores):
        result = {
            'image_name': image_name,
            'detections': []
        }
        for box, label, score in zip(boxes, labels, scores):
            detection = {
                'bbox': box.tolist(),    # [xmin, ymin, xmax, ymax]
                'label': int(label),     # Ensure it's serializable
                'score': float(score)
            }
            result['detections'].append(detection)
        
        self.results.append(result)

    def save_results_json(self, filename='results.json'):
        json_path = os.path.join(self.output_dir, filename)
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        print(f"[INFO] Detection results saved to {json_path}")

    def save_annotated_image(self, image_path, boxes, labels, scores, label_map=None, threshold=0.5):
        #image with drawn bounding boxes
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image {image_path} not found.")
        
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue  # Skip low-confidence detections
            
            color = (0, 255, 0)  # Green box
            thickness = 2
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

            label_text = f"{label}"
            if label_map and label in label_map:
                label_text = label_map[label]

            cv2.putText(
                image, 
                f"{label_text} {score:.2f}", 
                (x_min, y_min - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )

        img_name = os.path.basename(image_path)
        save_path = os.path.join(self.annotated_dir, img_name)
        cv2.imwrite(save_path, image)
        print(f"[INFO] Annotated image saved at {save_path}")