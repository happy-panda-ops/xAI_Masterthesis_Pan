import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from utils.file_os import generate_output_filename  # Import filename generator

class ObjectDetection:
    def __init__(self, det_model_path, det_input, det_output):
        self.det_model_path = det_model_path
        self.det_input = det_input
        self.det_output = det_output
        self.model = self.load_model()
        
        # Ensure output directory exists
        ImageOS.ensure_dir_exists(self.det_output)

    def load_model(self):
        """Load YOLO model"""
        return YOLO(self.det_model_path).to('cpu')

    def detect_objects(self, img):
        """Get object detection results"""
        try:
            results = self.model.predict(img)
            return results
        except Exception as e:
            print(f"Model inference error on image: {e}")
            return None

    def process_image(self):
        """Process a single image and save detection results"""
        img = ImageOS.read_image(self.det_input)
        if img is None:
            return

        results = self.detect_objects(img)
        if results is None:
            return

        new_filename = generate_output_filename(self.det_input, '_det')
        image_results = {"file_name": os.path.basename(self.det_input), "instances": []}
        detected_boxes = []

        # Define random colors for each class
        yolo_classes = list(self.model.names.values())
        class_colors = {cls: np.random.randint(0, 256, size=3).tolist() + [255] for cls in yolo_classes}  # RGBA colors

        # Create a transparent image
        img_height, img_width = img.shape[:2]
        transparent_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)

        detections = results[0]

        for box in detections.boxes:
            confidence = float(box.conf.item())  # Ensure it's a Python float
            if confidence > 0.4:  # Detection threshold
                cls = int(box.cls.item())
                cls_name = yolo_classes[cls]
                color = class_colors[cls_name]

                xyxy = box.xyxy.cpu().numpy()[0]
                x1, y1, x2, y2 = map(int, xyxy)  # Convert to Python int
                detected_boxes.append((x1, y1, x2, y2))

                # Add detection box to JSON results
                image_results["instances"].append({
                    "class_id": cls,
                    "class_name": cls_name,
                    "confidence": confidence,
                    "box": [x1, y1, x2, y2]
                })

                # Draw bounding box on transparent image
                cv2.rectangle(transparent_img, (x1, y1), (x2, y2), color, 2)

        # Save results
        self.export_results(new_filename, transparent_img, image_results)

    def export_results(self, new_filename, transparent_img, image_results):
        """Save detection boxes on transparent image and results to JSON"""
        # Save JSON file
        JsonOS.save_json_with_extension(os.path.join(self.det_output, new_filename), image_results)

        # Save transparent PNG image
        ImageOS.save_image_as_png(os.path.join(self.det_output, new_filename), transparent_img)

    def run(self):
        """Main process: handle input image and save detection results"""
        ImageOS.ensure_dir_exists(self.det_output)
        self.process_image()

# Main function call
if __name__ == "__main__":
    det_model_path = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\src\models\detec\Mixed_YOLO_d1+d2+s1_test_1-best.pt"
    det_input_image = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\example\demo_tests\2_tran\M1_0025_tran_transformed.png"
    det_output_folder = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\example\demo_tests\3_det"

    # Initialize and run detection class
    detector = ObjectDetection(det_model_path, det_input_image, det_output_folder)
    detector.run()
