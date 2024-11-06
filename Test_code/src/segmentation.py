# main_segmentation.py
import os
import numpy as np
import cv2
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from utils.file_os import generate_output_filename
from ultralytics import YOLO  # YOLOv8 library

class ObjectSegmentation:
    def __init__(self, seg_model, seg_input, seg_output, model_type='detectron'):
        self.seg_model = seg_model
        self.seg_input = seg_input
        self.seg_output = seg_output

        self.model_handler = self.YOLOHandler(seg_model)

        # # Choose model handler based on type
        # if model_type == 'detectron':
        #     self.model_handler = self.Detectron2Handler(seg_model)
        # elif model_type == 'yolo':
        #     self.model_handler = self.YOLOHandler(seg_model)
        # else:
        #     raise ValueError("Unsupported model type. Choose 'detectron' or 'yolo'.")

    def get_segmentation(self):
        """Get segmentation model predictions using the chosen handler"""
        seg_img = ImageOS.read_image(self.seg_input)
        if seg_img is None:
            print(f"Error: Could not read image at {self.seg_input}")
            return None

        # Use the model handler to process the image and get results
        result_dict, pred_polygons = self.model_handler.predict(seg_img, self.seg_input)

        return pred_polygons, result_dict

    def process_image(self):
        """Process image and extract segmentation results"""
        pred_polygons, result_dict = self.get_segmentation()
        if pred_polygons is None or result_dict is None:
            print("Segmentation failed.")
            return

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
        transparent_img = ImageOS.draw_polygon(img_size, polygons)

        # Save transparent PNG image
        ImageOS.save_image_as_png(os.path.join(self.seg_output, new_filename), transparent_img)

    def run(self):
        """Main process: handle input image and save segmentation results"""
        ImageOS.ensure_dir_exists(self.seg_output)
        self.process_image()

    class Detectron2Handler:
        """Handles all Detectron2-specific configuration, loading, and prediction tasks."""
        
        def __init__(self, model_path, threshold=0.6, device="cuda"):
            self.model_path = model_path
            self.threshold = threshold
            self.device = device
            self.cfg = self._load_config()
            self.predictor = DefaultPredictor(self.cfg)

        def _load_config(self):
            """Load and configure Detectron2 settings"""
            cfg = get_cfg()
            cfg.merge_from_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
            cfg.MODEL.DEVICE = self.device
            return cfg

        def predict(self, image):
            """Run model prediction and return formatted results"""
            predictions = self.predictor(image)
            instances = predictions["instances"].to("cpu")
            
            pred_classes = instances.pred_classes.tolist()
            pred_boxes = instances.pred_boxes.tensor.tolist()
            pred_scores = instances.scores.tolist()
            
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
                "file_name": os.path.basename(image),
                "instances": detection_results
            }

            return result_dict, pred_polygons

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
    class YOLOHandler:
        """Handles all YOLO-specific configuration, loading, and prediction tasks."""
        
        def __init__(self, model_path, confidence_threshold=0.6):
            self.model = YOLO(model_path).to('cpu')  # Load YOLO model from specified path
            self.confidence_threshold = confidence_threshold

        def predict(self, image, image_path):
            """Run YOLO model prediction and return formatted results"""
            results = self.model.predict(image)
            detections = results[0]

            # Process detections
            detection_results = []
            pred_polygons = []
            for detection in detections.boxes:
                score = detection.conf.item()
                if score >= self.confidence_threshold:
                    cls = int(detection.cls.item())
                    box = detection.xyxy.cpu().numpy().tolist()[0]  # Bounding box as [x1, y1, x2, y2]

                    # YOLO does not provide mask-based segmentation, so boxes are used as polygons here
                    polygons = [[box[:2], [box[0], box[3]], box[2:], [box[2], box[1]]]]
                    pred_polygons.append(polygons)

                    instance_result = {
                        "class": cls,
                        "box": box,
                        "score": score,
                        "mask": [{"polygon": poly} for poly in polygons]
                    }
                    detection_results.append(instance_result)

            result_dict = {
                "file_name": os.path.basename(image_path),  # Use image_path here
                "instances": detection_results
            }
            
            print(results)
            
            return result_dict, pred_polygons

# Main function call
if __name__ == "__main__":
    seg_model_path = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\src\models\seg\YOLO-seg-test-3best.pt"
    seg_input_image = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\example\demo_tests\1_seg\Model_2_0004.jpeg"
    seg_output_folder = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\example\demo_tests\1_seg"

    # Initialize and run segmentation with YOLO
    segmenter = ObjectSegmentation(seg_model_path, seg_input_image, seg_output_folder, model_type='yolo')
    segmenter.run()
