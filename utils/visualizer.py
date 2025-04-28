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

            label_text = f"ID: {label}"
            if label_map and label in label_map:
                label_text = label_map.get(label, f"ID: {label}")

            # Display label with score
            cv2.putText(
                image, 
                f"{label_text} {score:.2f}", 
                (x_min, y_min - 10), 
                self.font, 
                self.font_scale, 
                self.font_color, 
                self.font_thickness
            )

        return image
    







'''
Visualizer: It's for immediate visual feedback for developers or anyone interacting with the model, so they can see what’s being detected and how.

ResultSaver: It's about recording and storing the results for future use, so we can audit, retrain, or simply review the output later.
'''