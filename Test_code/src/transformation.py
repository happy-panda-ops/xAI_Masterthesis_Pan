import os
import cv2
import numpy as np
import json
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from utils.file_os import generate_output_filename

class PolygonTransformer:
    def __init__(self, image_path, json_path, output_folder):
        self.image_path = image_path
        self.json_path = json_path
        self.output_folder = output_folder

    def run(self):
        """Main method to process the image and JSON, and save the transformed outputs."""
        # Ensure output directory exists
        ImageOS.ensure_dir_exists(self.output_folder)

        # Load image
        image = ImageOS.read_image(self.image_path)
        if image is None:
            print(f"Error: Could not read image at {self.image_path}")
            return

        # Load JSON
        outputs = JsonOS.load_json(self.json_path)
        if outputs is None:
            print(f"Error: Could not read JSON file at {self.json_path}")
            return

        # Select the instance with the maximum area
        max_area_instance = self.select_max(outputs)

        if max_area_instance is None:
            print("No instances found in JSON.")
            return

        # Simplify the polygon to 4 corners
        simplified_polygon = self.simplify_polygon(
            max_area_instance['mask'][0]['polygon'], method="dp", epsilon_factor=0.02
        )

        # Ensure the simplified polygon has 4 points
        if len(simplified_polygon) != 4:
            print("Simplified polygon does not have 4 points.")
            return

        # Compute the perspective transformation matrix based on the simplified polygon
        M = self.compute_perspective_transform(simplified_polygon)

        # Apply the transformation matrix to the entire image and get the adjusted matrix
        transformed_image, M_adjusted = self.apply_transformation(image, M)

        # Apply the adjusted transformation matrix to the simplified polygon
        transformed_polygon = cv2.perspectiveTransform(
            simplified_polygon.reshape(-1, 1, 2).astype(np.float32), M_adjusted
        ).reshape(-1, 2)

        # Generate the transparent image with the transformed polygon
        polygon_overlay = self.create_transparent_polygon_image(transformed_image.shape, transformed_polygon)

        # Generate output filenames
        new_filename = generate_output_filename(self.image_path, '_tran')

        # Save the transformed image
        ImageOS.save_image_as_png(os.path.join(self.output_folder, new_filename + '_transformed'), transformed_image)

        # Save the transparent image with the transformed polygon
        ImageOS.save_image_as_png(os.path.join(self.output_folder, new_filename + '_polygon'), polygon_overlay)

        # Prepare the JSON data with original and transformed polygons
        result_dict = {
            "file_name": os.path.basename(self.image_path),
            "original_polygon": simplified_polygon.tolist(),
            "transformed_polygon": transformed_polygon.tolist(),
            "transformation_matrix": M.tolist()
        }
        # Save the new JSON
        JsonOS.save_json_with_extension(os.path.join(self.output_folder, new_filename), result_dict)

    def select_max(self, outputs):
        """Selects the instance with the maximum area from the JSON data."""
        max_area = 0
        max_area_instance = None
        for instance in outputs['instances']:
            for polygon_data in instance['mask']:
                polygon = polygon_data['polygon']
                area = self.polygon_area(polygon)
                if area > max_area:
                    max_area = area
                    max_area_instance = {
                        "class": instance["class"],
                        "box": instance["box"],
                        "score": instance["score"],
                        "mask": [{"polygon": polygon}]
                    }
        return max_area_instance

    @staticmethod
    def polygon_area(polygon):
        """Calculates the area of a polygon."""
        polygon = np.array(polygon)
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def simplify_polygon(self, polygon, method="dp", epsilon_factor=0.02):
        """Simplifies a polygon to a quadrilateral using the specified method."""
        if method == "dp":
            return self.dp_simplify(polygon, epsilon_factor)
        else:
            return polygon

    def dp_simplify(self, polygon, epsilon_factor=0.02):
        """Simplifies a polygon using the Douglas-Peucker algorithm."""
        contour = np.array(polygon, dtype=np.float32).reshape((-1, 1, 2))
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        simplified_polygon = approx.reshape(-1, 2)
        return simplified_polygon

    def compute_perspective_transform(self, simplified_polygon):
        """Computes the perspective transformation matrix based on the simplified polygon."""
        # Ensure vertices are in the correct order
        src_pts = self.order_points(simplified_polygon).astype(np.float32)

        # Define the destination points (mapped to a rectangle)
        widthA = np.linalg.norm(src_pts[2] - src_pts[3])
        widthB = np.linalg.norm(src_pts[1] - src_pts[0])
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(src_pts[1] - src_pts[2])
        heightB = np.linalg.norm(src_pts[0] - src_pts[3])
        maxHeight = max(int(heightA), int(heightB))

        dst_pts = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        # Compute the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return M

    def apply_transformation(self, image, M):
        """Applies the given transformation matrix to the entire image and returns the adjusted matrix."""
        # Get the size of the input image
        img_height, img_width = image.shape[:2]

        # Transform the image corners to find the size of the output image
        corners = np.array([
            [0, 0],
            [img_width - 1, 0],
            [img_width - 1, img_height - 1],
            [0, img_height - 1]
        ], dtype="float32")
        transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M)
        transformed_corners = transformed_corners.reshape(-1, 2)

        # Find the bounding rectangle of the transformed image
        x_coords = transformed_corners[:, 0]
        y_coords = transformed_corners[:, 1]
        x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
        y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))

        # Compute the size of the output image
        output_width = x_max - x_min
        output_height = y_max - y_min

        # Adjust the transformation matrix to account for translation
        translation_matrix = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ], dtype=np.float32)
        M_adjusted = translation_matrix @ M

        # Apply the adjusted transformation to the entire image
        transformed_image = cv2.warpPerspective(
            image, M_adjusted, (output_width, output_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        return transformed_image, M_adjusted

    def create_transparent_polygon_image(self, image_shape, polygon):
        """Creates a transparent image with the transformed polygon contour drawn on it."""
        # Create an empty image with 4 channels (RGBA)
        transparent_img = np.zeros((image_shape[0], image_shape[1], 4), dtype=np.uint8)

        # Define the color with full opacity
        color = (0, 255, 0, 255)  # Green color with full opacity

        # Convert polygon points to integer
        points = np.array([polygon], dtype=np.int32)

        # Draw the polygon contour on the transparent image
        cv2.polylines(transparent_img, [points], isClosed=True, color=color, thickness=2)

        return transparent_img


    @staticmethod
    def order_points(pts):
        """Orders points in the order: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]     # Top-left has the smallest sum
        rect[2] = pts[np.argmax(s)]     # Bottom-right has the largest sum

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right has the smallest difference
        rect[3] = pts[np.argmax(diff)]  # Bottom-left has the largest difference

        return rect


# Main block to run the transformer independently
if __name__ == "__main__":
    # Define the input image path, input JSON path, and output folder
    image_path = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\example\demo_tests\1_seg\M1_0025.jpeg"  # Replace with your image path
    segmentation_json_path = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\example\demo_tests\1_seg\M1_0025_seg.json"     # Replace with your JSON path
    output_folder = r"C:\Users\ba7jd2\Work Folders\Documents\Projects\WoodFeature\xAI_Masterthesis_Pan\Test_code\example\demo_tests\2_tran"            # Replace with your output folder

    transformer = PolygonTransformer(image_path, segmentation_json_path, output_folder)
    transformer.run()