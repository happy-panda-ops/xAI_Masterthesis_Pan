# json_os.py
import os
import json

class JsonOS:
    @staticmethod
    def ensure_dir_exists(directory):
        """Ensure directory exists, create if it doesn't"""
        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def save_json_with_extension(path, data):
        """Save data to JSON file with .json extension"""
        JsonOS.ensure_dir_exists(os.path.dirname(path))
        json_path = f"{path}.json"  # Add .json extension
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Results saved as JSON: {json_path}")

    @staticmethod
    def load_json(input_path):
        """Load JSON data from a file."""
        if not os.path.exists(input_path):
            print(f"JSON file does not exist: {input_path}")
            return None
        with open(input_path, 'r') as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def save_json_with_extension(output_path, data):
        """Save JSON data to a file with .json extension."""
        output_path = os.path.splitext(output_path)[0]  # Remove any existing extension
        with open(output_path + '.json', 'w') as f:
            json.dump(data, f, indent=4)
