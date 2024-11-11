# otsu_filter.py
import os
import cv2
import json
import numpy as np
from utils.image_os import ImageOS
from utils.json_os import JsonOS
from file_utils import generate_output_filename
from PIL import Image, ImageDraw

    
class kNNFilter:
    
    def __init__(self, input_image_path, input_json_path, output_path):
        self.input_image_path = input_image_path
        self.input_json_path = input_json_path
        self.output_path = output_path
        ImageOS.ensure_dir_exists(self.output_path)  # Ensure output directory exists

    
    # Define transformation for the images
    def get_transform():
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load a pre-trained ResNet-18 model
    def get_pretrained_resnet18():
        model = resnet18(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fully connected layer
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device)

    # Extract patches and features from the image based on bounding boxes
    def extract_patches_and_features(image, bboxes, model, transform, device):
        patches = []
        features = []

        for bbox in bboxes:
            # Extract patch
            x_min, y_min, x_max, y_max = bbox
            patch = image[y_min:y_max, x_min:x_max]
            patch = Image.fromarray(patch)
            
            # Preprocess patch
            transformed_patch = transform(patch).unsqueeze(0).to(device)
            
            # Extract feature using the model
            with torch.no_grad():
                feature = model(transformed_patch).squeeze().cpu().numpy()
            
            patches.append(transformed_patch)
            features.append(feature)
        
        return patches, features

    # Load YOLO annotations (bounding boxes)
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

    # Read JSON file
    def read_json(json_file):
        with open(json_file, 'r') as file:
            return json.load(file)

    def save_image_with_bboxes(image, bboxes, scores, save_path, font_size=20, font_color=(255, 0, 0), font_thickness=2):
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


    # KNN class for cosine similarity-based search
    class cos_kNN:
        def __init__(self, k):
            self.k = k
            self.X_train = None

        def fit(self, X_train):
            self.X_train = X_train

        def predict(self, X_test):
            # Normalize both X_train and X_test -> unit vectors
            X_train_norm = torch.nn.functional.normalize(self.X_train, p=2, dim=1)
            X_test_norm = torch.nn.functional.normalize(X_test, p=2, dim=1)

            # Cosine similarity and distances
            cosine_similarity = torch.matmul(X_test_norm, X_train_norm.T)
            cosine_distance = 1 - cosine_similarity

            # Find k nearest neighbors based on cosine distance
            knn_indices = cosine_distance.topk(self.k, dim=1, largest=False).indices
            return knn_indices

    # Main function to process images, extract features, and perform kNN search
    def process_knn(image_folder, target_folder, output_folder, knn_bank, distance_threshold=0.3):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transform = get_transform()
        model = get_pretrained_resnet18()

        # Load the kNN feature bank
        all_data = np.load(knn_bank)
        all_features = all_data['features']
        all_features_tensor = torch.tensor(all_features, dtype=torch.float32)

        # Traverse target folder to find all JSON files
        for target_file in os.listdir(target_folder):
            if target_file.endswith('.json'):
                json_path = os.path.join(target_folder, target_file)
                json_data = read_json(json_path)

                # Get image name from JSON and load image
                image_name = json_data['image']
                img_path = os.path.join(image_folder, image_name)

                if not os.path.exists(img_path):
                    print(f"Image {image_name} not found in {image_folder}, skipping...")
                    continue

                # Read image and convert to RGB
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                target_bboxes = []
                target_features = []
                target_scores = []
                filtered_results = []

                # Extract bounding boxes and confidence scores from JSON
                for box in json_data['boxes']:
                    coordinates = [box['x1'], box['y1'], box['x2'], box['y2']]
                    target_bboxes.append(coordinates)
                    target_scores.append(box['confidence'])


                # Extract patches and features
                patches, features = extract_patches_and_features(image, target_bboxes, model, transform, device)
                target_features.extend(features)

                # KNN search for each target feature
                for index, target_feature in enumerate(target_features):
                    target_feature_tensor = torch.tensor(target_feature, dtype=torch.float32).unsqueeze(0)

                    # Fit kNN model
                    knn = cos_kNN(k=10)
                    knn.fit(all_features_tensor)

                    # Predict neighbors
                    predicted_neighbors = knn.predict(target_feature_tensor)

                    # Calculate cosine similarity for k nearest neighbors
                    X_train_norm = torch.nn.functional.normalize(all_features_tensor[predicted_neighbors].squeeze(0), p=2, dim=1)
                    X_test_norm = torch.nn.functional.normalize(target_feature_tensor, p=2, dim=1)
                    cosine_similarity = torch.nn.functional.cosine_similarity(X_test_norm, X_train_norm)
                    cosine_distance = 1 - cosine_similarity

                    # Filter results based on distance threshold
                    if cosine_distance.min().item() < distance_threshold:
                        filtered_results.append({
                            "target_index": index,
                            "target_bbox": target_bboxes[index],
                            "predicted_neighbors": predicted_neighbors.cpu().numpy().tolist(),
                            "min_distance": cosine_distance.min().item()
                        })

                # Save the filtered results with the knn_red_nms_ prefix
                output_json_name = f"knn_red_nms_{os.path.splitext(image_name)[0]}.json"
                output_json_path = os.path.join(output_folder, output_json_name)

                if filtered_results:
                    save_json_with_bboxes(json_data, target_bboxes, target_scores, output_json_path, image_name)

                    # Save the image with bounding boxes
                    output_img_name = f"knn_red_nms_{image_name}"
                    output_img_path = os.path.join(output_folder, output_img_name)
                    save_image_with_bboxes(image, target_bboxes, target_scores, output_img_path, font_size=1, font_color=(255, 0, 0), font_thickness=4)
                
                print(f"Processed KNN results saved to: {output_json_path}")

    # Paths for input, output and kNN bank
    image_folder = "/Users/holmes/Documents/UNI-Bamberg/Arbeiten/Datensatz/Selfmade/4_Maria_0710/Test_general_workflow/5_trans"
    target_folder = "/Users/holmes/Documents/UNI-Bamberg/Arbeiten/Datensatz/Selfmade/4_Maria_0710/Test_general_workflow/5_trans/3_ostu"
    output_folder = "/Users/holmes/Documents/UNI-Bamberg/Arbeiten/Datensatz/Selfmade/4_Maria_0710/Test_general_workflow/5_trans/4_knn"
    knn_bank = '/Users/holmes/Documents/UNI-Bamberg/4.Semester_MA/Masterthesis/xAI_Masterthesis_Pan/Code/Bias/kNN_search/kNN_knot.npz'

    # Run the process
    process_knn(image_folder, target_folder, output_folder, knn_bank)
