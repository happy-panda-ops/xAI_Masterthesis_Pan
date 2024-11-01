# main_detection.py
import os
from ultralytics import YOLO
from image_os import ImageOS
from json_os import JsonOS
from file_utils import generate_output_filename  # Import filename generator

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
        return YOLO(self.det_model_path)

    def detect_objects(self, img):
        """Get object detection results"""
        try:
            return self.model(img)
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
        image_results = {"image": os.path.basename(self.det_input), "boxes": []}
        detected_boxes = []

        for result in results:
            if not result.boxes:
                continue

            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence > 0.4:  # Detection threshold
                    xyxy = box.xyxy[0].astype(int)
                    x1, y1, x2, y2 = xyxy
                    detected_boxes.append((x1, y1, x2, y2))

                    # Add detection box to JSON results
                    image_results["boxes"].append({
                        "confidence": confidence,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })

        # Save results
        self.export_results(new_filename, detected_boxes, image_results)

    def export_results(self, new_filename, boxes, image_results):
        """Save detection boxes on transparent image and results to JSON"""
        # Save JSON file
        JsonOS.save_json_with_extension(os.path.join(self.det_output, new_filename), image_results)

        # Draw detection boxes on transparent background and save as PNG
        img = ImageOS.read_image(self.det_input)
        img_size = img.shape[:2]  # Get image dimensions (height, width)
        transparent_img = ImageOS.draw_boxes_on_transparent_background(img_size, boxes)
        
        # Save transparent PNG image
        ImageOS.save_image_as_png(os.path.join(self.det_output, new_filename), transparent_img)

    def run(self):
        """Main process: handle input image and save detection results"""
        ImageOS.ensure_dir_exists(self.det_output)
        self.process_image()

# Main function call
if __name__ == "__main__":
    det_model_path = "path/to/your/best.pt"
    det_input_image = "path/to/input/image"
    det_output_folder = "path/to/output/folder"

    # Initialize and run detection class
    detector = ObjectDetection(det_model_path, det_input_image, det_output_folder)
    detector.run()
