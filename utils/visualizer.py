import cv2

class Visualizer:
    def __init__(self, color=(0, 255, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_color=(0, 255, 0), font_thickness=2):
        self.color = color
        self.thickness = thickness
        self.font = font
        self.font_scale = font_scale
        self.font_color = font_color
        self.font_thickness = font_thickness

    def draw_boxes(self, image, boxes, labels, scores, label_map=None, threshold=0.5):
        #Draw bounding boxes and labels on the image.
        for box, label, score in zip(boxes, labels, scores):
            if score < threshold:
                continue  # Skip low-confidence detections
          
            x_min, y_min, x_max, y_max = map(int, box)
            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), self.color, self.thickness)

            label_text = f"{COCO_INSTANCE_CATEGORY_NAMES[label]} {score:.2f}"

            # Display label with score
            cv2.putText(
                image, 
                label_text, 
                (x_min, y_min - 10), 
                self.font, 
                self.font_scale, 
                self.font_color, 
                self.font_thickness
            )

        return image
    

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
    # ... add all 91 class names if needed
]







'''
Visualizer: It's for immediate visual feedback for developers or anyone interacting with the model, so they can see what’s being detected and how.

ResultSaver: It's about recording and storing the results for future use, so we can audit, retrain, or simply review the output later.
'''