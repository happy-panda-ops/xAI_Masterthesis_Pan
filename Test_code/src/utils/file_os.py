# file_utils.py
import os

def generate_output_filename(input_path, suffix):
    """
    Generate output filename by adding a suffix to the original filename.
    Example: "image.jpg" -> "image_det.jpg" or "image_seg.jpg"
    
    Args:
        input_path (str): Original input file path.
        suffix (str): Suffix to add (e.g., '_det' or '_seg').
    
    Returns:
        str: Generated output filename with suffix.
    """
    base, ext = os.path.splitext(os.path.basename(input_path))
    return f"{base}{suffix}{ext}"
