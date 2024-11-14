import os
import cv2
import numpy as np
from ultralytics import YOLO
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from utils.file_os import generate_output_filename  # Import filename generator

class YOLOHandler:
    """Handles all YOLO-specific configuration, loading, and prediction tasks for detection."""

    def __init__(self, model_path, confidence_threshold=0.4):
        self.model = YOLO(model_path).to('cpu')  # Load YOLO model from specified path
        self.confidence_threshold = confidence_threshold
        self.yolo_classes = list(self.model.names.values())
        # Define random colors for each class (RGBA)
        self.class_colors = {cls: np.random.randint(0, 256, size=3).tolist() + [255] for cls in self.yolo_classes}

    def predict(self, image):
        """Run YOLO model prediction and return formatted results and transparent image."""
        try:
            results = self.model.predict(image)
            detections = results[0]

            # Create a transparent image
            img_height, img_width = image.shape[:2]
            transparent_img = np.zeros((img_height, img_width, 4), dtype=np.uint8)

            image_results = {"instances": []}

            for box in detections.boxes:
                confidence = float(box.conf.item())
                if confidence > self.confidence_threshold:
                    cls = int(box.cls.item())
                    cls_name = self.yolo_classes[cls]
                    color = self.class_colors[cls_name]

                    xyxy = box.xyxy.cpu().numpy()[0]
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Add detection box to results
                    image_results["instances"].append({
                        "class_id": cls,
                        "class_name": cls_name,
                        "confidence": confidence,
                        "box": [x1, y1, x2, y2]
                    })

                    # Draw bounding box on transparent image
                    cv2.rectangle(transparent_img, (x1, y1), (x2, y2), color, 2)

            return image_results, transparent_img
        except Exception as e:
            print(f"Model inference error on image: {e}")
            return None, None

class ObjectDetection:
    def __init__(self, det_model_path, det_input, det_output):
        self.det_model_path = det_model_path
        self.det_input = det_input
        self.det_output = det_output

        # Initialize the model handler
        self.model_handler = YOLOHandler(det_model_path)

        # Ensure output directory exists
        ImageOS.ensure_dir_exists(self.det_output)

    def process_image(self):
        """Process a single image and save detection results."""
        img = ImageOS.read_image(self.det_input)
        if img is None:
            print(f"Error: Could not read image at {self.det_input}")
            return

        image_results, transparent_img = self.model_handler.predict(img)
        if image_results is None:
            print("Detection failed.")
            return

        new_filename = generate_output_filename(self.det_input, '_det')
        print(new_filename)
        image_results["file_name"] = os.path.basename(self.det_input)

        # Save results
        self.export_results(image_results, new_filename, transparent_img)

    def export_results(self, image_results, new_filename, transparent_img):
        """Save detection boxes on transparent image and results to JSON."""
        # Save JSON file
        JsonOS.save_json_with_extension(os.path.join(self.det_output, new_filename), image_results)

        # Save transparent PNG image
        ImageOS.save_image_as_png(os.path.join(self.det_output, new_filename), transparent_img)

    def run(self):
        """Main process: handle input image and save detection results."""
        ImageOS.ensure_dir_exists(self.det_output)
        self.process_image()

# Main function call
if __name__ == "__main__":
    det_model_path = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/src/models/detec/Mixed_YOLO_d1+d2+s1_test_1-best.pt"
    det_input = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/example/demo_tests/2_tran/M1_0025_tran_transformed.png"
    det_output = r"/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Test_code/example/demo_tests/3_det"

    # Initialize and run detection class
    detector = ObjectDetection(det_model_path, det_input, det_output)
    detector.run()
