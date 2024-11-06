# file_utils.py
import os

def generate_output_filename(input_path, suffix):
    """Generate output filename by adding a suffix before the file extension."""
    # Extract the base filename without the directory
    base_name = os.path.basename(input_path)
    # Split the filename into name and extension
    name_without_ext, _ = os.path.splitext(base_name)
    # Add the suffix to the filename without extension
    new_filename = name_without_ext + suffix
    return new_filename
