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
    def read_json(path):
        """Read JSON file"""
        try:
            with open(path, 'r') as json_file:
                return json.load(json_file)
        except Exception as e:
            print(f"Error reading JSON file {path}: {e}")
            return None

    @staticmethod
    def save_json_with_extension(path, data):
        """Save data to JSON file with .json extension"""
        JsonOS.ensure_dir_exists(os.path.dirname(path))
        json_path = f"{path}.json"  # Add .json extension
        with open(json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Results saved as JSON: {json_path}")
