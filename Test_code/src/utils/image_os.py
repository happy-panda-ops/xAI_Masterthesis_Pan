# image_os.py
import os
import cv2
import numpy as np

class ImageOS:
    @staticmethod
    def ensure_dir_exists(directory):
        """Ensure directory exists, create if it doesn't"""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def read_image(img_path):
        """Read image file"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error reading image: {img_path}")
        return img

    @staticmethod
    def save_image_as_png(path, img):
        """Save image as PNG format"""
        ImageOS.ensure_dir_exists(os.path.dirname(path))
        png_path = f"{path}.png"  # Add .png extension
        cv2.imwrite(png_path, img)
        print(f"Image saved as PNG: {png_path}")

    @staticmethod
    def draw_polygon(img_size, polygons, color=(0, 255, 0), thickness=2):
        """Draw polygons on a transparent background"""
        transparent_img = np.zeros((img_size[0], img_size[1], 4), dtype=np.uint8)
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32)
            cv2.polylines(transparent_img, [points], isClosed=True, color=color + (255,), thickness=thickness)
        return transparent_img

    @staticmethod
    def draw_boxes(img_size, boxes, color=(0, 0, 255), thickness=2):
        """Draw bounding boxes on a transparent background"""
        transparent_img = np.zeros((img_size[0], img_size[1], 4), dtype=np.uint8)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(transparent_img, (x1, y1), (x2, y2), color + (255,), thickness=thickness)
        return transparent_img
