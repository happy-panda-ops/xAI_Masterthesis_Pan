# otsu_filter.py
import os
import cv2
import json
import numpy as np
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from file_utils import generate_output_filename
from PIL import Image, ImageDraw

class OtsuFilter:
    
    def __init__(self, input_image_path, input_json_path, output_path):
        self.input_image_path = input_image_path
        self.input_json_path = input_json_path
        self.output_path = output_path
        ImageOS.ensure_dir_exists(self.output_path)  # Ensure output directory exists

    def get_otsu_threshold(self, img):
        """Calculate Otsu threshold for the image patch"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_thresh = cv2.bitwise_not(otsu_thresh)  # Invert to get white background
        return otsu_thresh

    def apply_otsu_filter_on_patch(self, patch):
        """Apply Otsu filter on a single patch and find largest contour bounding box"""
        otsu_thresh = self.get_otsu_threshold(patch)
        contours, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return [x, y, x + w, y + h]
        return None

    def adjust_bboxes_with_otsu(self, image, bboxes):
        """Adjust each bounding box in the image using Otsu threshold"""
        adjusted_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            patch = image[y_min:y_max, x_min:x_max]
            adjusted_bbox = self.apply_otsu_filter_on_patch(patch)

            if adjusted_bbox:
                # Adjust to the original image coordinates
                new_x_min = x_min + adjusted_bbox[0]
                new_y_min = y_min + adjusted_bbox[1]
                new_x_max = x_min + adjusted_bbox[2]
                new_y_max = y_min + adjusted_bbox[3]
                adjusted_bboxes.append([new_x_min, new_y_min, new_x_max, new_y_max])
            else:
                adjusted_bboxes.append(bbox)  # If no adjustment, keep original bbox
        return adjusted_bboxes

    def process_image(self):
        """Process image and JSON file, apply Otsu filter on bounding boxes, and save results"""
        image = ImageOS.read_image(self.input_image_path)
        if image is None:
            return

        json_data = JsonOS.read_json(self.input_json_path)
        if json_data is None or "boxes" not in json_data:
            print("Invalid JSON format")
            return

        # Extract original bounding boxes
        original_bboxes = [[box['x1'], box['y1'], box['x2'], box['y2']] for box in json_data["boxes"]]
        scores = [box['confidence'] for box in json_data["boxes"]]

        # Adjust bounding boxes using Otsu threshold
        adjusted_bboxes = self.adjust_bboxes_with_otsu(image, original_bboxes)

        # Generate output filename without extension
        output_filename = generate_output_filename(self.input_image_path, '_otsu')

        # Save adjusted bounding boxes to transparent PNG
        transparent_img = self.draw_bboxes_on_transparent(image.shape[:2], adjusted_bboxes)
        ImageOS.save_image_as_png(os.path.join(self.output_path, output_filename), transparent_img)

        # Save adjusted bounding boxes and scores to JSON
        updated_json_data = {
            "image": os.path.basename(self.input_image_path),
            "boxes": [{"confidence": score, "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}
                      for box, score in zip(adjusted_bboxes, scores)]
        }
        JsonOS.save_json_with_extension(os.path.join(self.output_path, output_filename), updated_json_data)

    def draw_bboxes_on_transparent(self, img_size, bboxes, color=(255, 0, 0), thickness=2):
        """Draw bounding boxes on a transparent background"""
        transparent_img = np.zeros((img_size[0], img_size[1], 4), dtype=np.uint8)
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(transparent_img, (x_min, y_min), (x_max, y_max), color + (255,), thickness=thickness)
        return transparent_img

    def run(self):
        """Main process: handle input image and JSON file, apply Otsu filter, save results"""
        self.process_image()

# Main function call example
if __name__ == "__main__":
    input_image = "/path/to/input/image.jpg"
    input_json = "/path/to/input/annotations.json"
    output_folder = "/path/to/output/folder"

    # Initialize and run Otsu filter processing
    otsu_processor = OtsuFilter(input_image, input_json, output_folder)
    otsu_processor.run()