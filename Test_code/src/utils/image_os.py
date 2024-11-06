# image_os.py
import os
import cv2
import numpy as np

class ImageOS:
    @staticmethod
    def ensure_dir_exists(directory):
        """Ensure directory exists, create if it doesn't."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def read_image(img_path):
        """Read an image from a file."""
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Error reading image: {img_path}")
        return img

    @staticmethod
    def save_image_as_png(output_path, image):
        """Save image as PNG, ensuring correct handling of transparency if present."""
        output_path = os.path.splitext(output_path)[0]  # Remove any existing extension
        cv2.imwrite(output_path + '.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    @staticmethod
    def draw_polygon(img_size, polygons, color=(0, 255, 0, 255), thickness=-1):
        """Draw polygons on a transparent background."""
        # Create a transparent image
        transparent_img = np.zeros((img_size[0], img_size[1], 4), dtype=np.uint8)
        # Ensure color has four components (RGBA)
        if len(color) == 3:
            color = color + (255,)  # Add full opacity if alpha not specified
        for polygon in polygons:
            points = np.array(polygon, dtype=np.int32)
            cv2.fillPoly(transparent_img, [points], color)
        return transparent_img

    @staticmethod
    def draw_boxes(img_size, boxes, color=(0, 0, 255, 255), thickness=2):
        """Draw bounding boxes on a transparent background."""
        # Create a transparent image
        transparent_img = np.zeros((img_size[0], img_size[1], 4), dtype=np.uint8)
        # Ensure color has four components (RGBA)
        if len(color) == 3:
            color = color + (255,)  # Add full opacity if alpha not specified
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(transparent_img, (x1, y1), (x2, y2), color, thickness=thickness)
        return transparent_img
