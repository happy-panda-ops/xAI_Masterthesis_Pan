# main_segmentation.py
import os
import numpy as np
import cv2
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from file_utils import generate_output_filename  # Import filename generator

class ObjectSegmentation:
    def __init__(self, seg_model, seg_input, seg_output):
        self.seg_model = seg_model
        self.seg_input = seg_input
        self.seg_output = seg_output

    def get_segmentation(self):
        """Get segmentation model predictions"""
        seg_img = ImageOS.read_image(self.seg_input)
        if seg_img is None:
            return None

        # Load model weights and set threshold (example)
        cfg.MODEL.WEIGHTS = '/Users/holmes/Desktop/test_10_0610_model_final.pth'  
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
        predictor = DefaultPredictor(cfg)
        
        predictions = predictor(seg_img)
        return predictions

    @staticmethod
    def binary_mask_to_polygons(binary_mask, tolerance=1):
        """Convert binary mask to polygons"""
        polygons = []
        binary_mask = binary_mask.astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 4:
                contour = contour.flatten().tolist()
                polygons.append([[contour[i], contour[i + 1]] for i in range(0, len(contour), 2)])
        return polygons

    def process_image(self):
        """Process image and extract segmentation results"""
        predictions = self.get_segmentation()
        if predictions is None:
            return

        instances = predictions["instances"].to("cpu")
        pred_classes = instances.pred_classes.tolist()
        pred_boxes = instances.pred_boxes.tensor.tolist()
        pred_scores = instances.scores.tolist()
        
        # Convert binary masks to polygons
        pred_masks = instances.pred_masks.numpy()
        pred_polygons = [self.binary_mask_to_polygons(mask) for mask in pred_masks]

        detection_results = []
        for cls, box, score, polygons in zip(pred_classes, pred_boxes, pred_scores, pred_polygons):
            instance_result = {
                "class": cls,
                "box": box,
                "score": score,
                "mask": [{"polygon": polygon} for polygon in polygons]
            }
            detection_results.append(instance_result)

        result_dict = {
            "file_name": os.path.basename(self.seg_input),
            "instances": detection_results
        }

        # Generate filename without extension and save results
        new_filename = generate_output_filename(self.seg_input, '_seg')
        self.export_results(result_dict, pred_polygons, new_filename)

    def export_results(self, result_dict, polygons, new_filename):
        """Save segmentation results as JSON and transparent PNG"""
        # Save JSON file
        JsonOS.save_json_with_extension(os.path.join(self.seg_output, new_filename), result_dict)

        # Create polygon overlay on transparent background
        img = ImageOS.read_image(self.seg_input)
        img_size = img.shape[:2]  # Get image dimensions (height, width)
        transparent_img = ImageOS.draw_polygon_on_transparent_background(img_size, polygons)

        # Save transparent PNG image
        ImageOS.save_image_as_png(os.path.join(self.seg_output, new_filename), transparent_img)

    def run(self):
        """Main process: handle input image and save segmentation results"""
        ImageOS.ensure_dir_exists(self.seg_output)
        self.process_image()

# Main function call
if __name__ == "__main__":
    seg_model_path = "path/to/your/segmentation_model"
    seg_input_image = "path/to/your/seg_input_image"
    seg_output_folder = "path/to/output/folder"

    # Initialize and run segmentation class
    segmenter = ObjectSegmentation(seg_model_path, seg_input_image, seg_output_folder)
    segmenter.run()
