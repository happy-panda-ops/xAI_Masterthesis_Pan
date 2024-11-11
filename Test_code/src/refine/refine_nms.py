# otsu_filter.py
import os
import cv2
import json
import numpy as np
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from file_utils import generate_output_filename
from PIL import Image, ImageDraw
    
class NMSFilter:

    def __init__(self, input_image_path, input_json_path, output_path):
        self.input_image_path = input_image_path
        self.input_json_path = input_json_path
        self.output_path = output_path
        ImageOS.ensure_dir_exists(self.output_path)  # Ensure output directory exists


    def load_yolo_annotations(annotation_file, image):
        bboxes = []
        with open(annotation_file, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                x_center, y_center, width, height = map(float, parts[1:5])
                img_width, img_height = image.shape[1], image.shape[0]
                x_min = int((x_center - width / 2) * img_width)
                y_min = int((y_center - height / 2) * img_width)
                x_max = int((x_center + width / 2) * img_width)
                y_max = int((y_center + height / 2) * img_width)
                bboxes.append([x_min, y_min, x_max, y_max])
        return bboxes

    def read_json(json_file):
        with open(json_file, 'r') as file:
            return json.load(file)

    def extract_patches(image, bboxes):
        patches = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            patch = image[y_min:y_max, x_min:x_max]
            patch = Image.fromarray(patch)
            patches.append(patch)
        return patches

    def non_max_suppression(boxes, scores, iou_threshold):
        if boxes.size(0) == 0:
            return torch.empty((0,), dtype=torch.int64)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        _, order = scores.sort(0, descending=True)
        keep = []
        while order.numel() > 0:
            i = order[0]
            keep.append(i)

            if order.numel() == 1:
                break

            xx1 = torch.max(x1[i], x1[order[1:]])
            yy1 = torch.max(y1[i], y1[order[1:]])
            xx2 = torch.min(x2[i], x2[order[1:]])
            yy2 = torch.min(y2[i], y2[order[1:]])

            w = torch.clamp(xx2 - xx1 + 1, min=0)
            h = torch.clamp(yy2 - yy1 + 1, min=0)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = torch.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return torch.tensor(keep, dtype=torch.int64)

    def save_image_with_bboxes(image, bboxes, scores, save_path, font_size=20, font_color=(0, 0, 255), font_thickness=4):
        """
        Save the image with bounding boxes and their corresponding confidence scores using OpenCV.

        Args:
        - image: The image on which to draw (in NumPy array format).
        - bboxes: A list of bounding boxes, each as (x1, y1, x2, y2).
        - scores: A list of confidence scores corresponding to each bounding box.
        - save_path: The path where the image with bounding boxes will be saved.
        - font_size: The size of the font for displaying confidence scores.
        - font_color: The color of the font for displaying confidence scores (BGR tuple).
        - font_thickness: The thickness of the font.
        """
        # Convert the PIL image to an OpenCV image (NumPy array)
        image = np.array(image)

        for bbox, score in zip(bboxes, scores):
            # Draw the rectangle for the bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(255, 0, 0), thickness=3)

            # Prepare the text (confidence score)
            text = f"{score:.2f}"
            
            # Set text position (top-left corner of the bounding box)
            text_position = (bbox[0], bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10)
            
            # Put the text on the image
            cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_thickness)

        # Save the image using OpenCV
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # Convert RGB back to BGR before saving
            
    def save_json_with_bboxes(json_data, boxes_tuples, scores_list, save_path, image_name):
        """
        Save the processed JSON file with updated image name and bounding boxes.

        Args:
        - json_data: The original JSON data.
        - boxes_tuples: A list of processed bounding boxes, each containing (x1, y1, x2, y2).
        - scores_list: A list of confidence scores corresponding to the bounding boxes.
        - save_path: The path where the updated JSON file will be saved.
        - image_name: The name of the processed image to be added in the JSON data.
        """
        # Add the processed image name to the JSON data
        json_data["image"] = image_name
        
        # Clear the existing bounding box data in the JSON and add the updated boxes and scores
        json_data["boxes"] = []
        
        # Iterate over each bounding box and its corresponding confidence score
        for box, score in zip(boxes_tuples, scores_list):
            # Append the bounding box and its confidence score to the JSON data
            json_data["boxes"].append({
                "confidence": score,
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3]
            })
        
        # Save the updated JSON data to the specified file path
        with open(save_path, 'w') as file:
            json.dump(json_data, file, indent=4)
            

    def process_folder(image_folder, target_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历target_folder中的所有json文件
        for target_file in os.listdir(target_folder):
            if target_file.endswith('.json'):
                # 读取JSON文件路径
                json_path = os.path.join(target_folder, target_file)
                json_data = read_json(json_path)

                # 从JSON数据中获取图像名称
                image_name = json_data['image']
                img_path = os.path.join(image_folder, image_name)

                # 如果图像不存在，跳过该文件
                if not os.path.exists(img_path):
                    print(f"Image {image_name} not found in {image_folder}, skipping...")
                    continue

                # 读取图像
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                original_image = Image.fromarray(image)

                target_bboxes = []
                target_scores = []

                # 从JSON中提取边界框和置信度
                for box in json_data['boxes']:
                    coordinates = (box['x1'], box['y1'], box['x2'], box['y2'])
                    target_bboxes.append(coordinates)
                    confidence = box['confidence']
                    target_scores.append(confidence)

                # 将边界框和置信度转换为tensor
                boxes = torch.tensor(target_bboxes, dtype=torch.float32)
                scores = torch.tensor(target_scores, dtype=torch.float32)
                iou_threshold = 0.3
                keep = non_max_suppression(boxes, scores, iou_threshold)

                # 过滤后的边界框和置信度
                filtered_boxes = boxes[keep]
                filtered_scores = scores[keep]

                # 转换为列表和元组
                filtered_boxes_list = filtered_boxes.tolist()
                filtered_scores_list = filtered_scores.tolist()
                filtered_boxes_tuples = [tuple(int(coordinate) for coordinate in box) for box in filtered_boxes_list]

                print(f"Kept boxes after NMS (as tuples): {filtered_boxes_tuples}")
                print(f"Scores after NMS: {filtered_scores_list}")

                # 保存处理后的图像和JSON文件，添加前缀nms_
                base_filename = os.path.splitext(image_name)[0]
                output_img_name = f"nms_{image_name}"
                output_img_path = os.path.join(output_folder, output_img_name)
                
                output_json_name = f"nms_{base_filename}.json"
                output_json_path = os.path.join(output_folder, output_json_name)
                
                save_image_with_bboxes(original_image, filtered_boxes_tuples, filtered_scores_list, output_img_path, font_size=1, font_color=(255, 0, 0), font_thickness=2)
                save_json_with_bboxes(json_data, filtered_boxes_tuples, filtered_scores_list, output_json_path, image_name)

                print(f"Processed image saved to: {output_img_path}")
                print(f"Processed JSON saved to: {output_json_path}")
                            
    # define input folder and output folder
    image_folder = "/Users/holmes/Documents/UNI-Bamberg/Arbeiten/Datensatz/Selfmade/4_Maria_0710/Test_general_workflow/5_trans"
    target_folder = f'/Users/holmes/Documents/UNI-Bamberg/Arbeiten/Datensatz/Selfmade/4_Maria_0710/Test_general_workflow/5_trans/1_detect'
    output_folder = f'/Users/holmes/Documents/UNI-Bamberg/Arbeiten/Datensatz/Selfmade/4_Maria_0710/Test_general_workflow/5_trans/2_nms'


process_folder(image_folder, target_folder, output_folder)