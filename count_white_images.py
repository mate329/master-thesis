import os
from PIL import Image
import numpy as np

def count_white_png_images(directory):
    """
    Recursively finds PNG files in the directory and counts those that are completely white.

    Parameters:
    directory (str): The path of the directory to search for PNG files.

    Returns:
    int: The count of completely white PNG images.
    """
    white_image_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png') and 'enterPINactivity' in file:
                file_path = os.path.join(root, file)
                
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        # Convert image to numpy array
                        img_array = np.array(img)
                        
                        # Check if the image is completely white
                        if np.all(img_array == 255):
                            white_image_count += 1
                            print(f"White image found: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

    return white_image_count

# Example usage
directory = './spectograms/spectrograms_enterPINactivity'  # Replace with the path to your directory
white_image_count = count_white_png_images(directory)
print(f"Total completely white PNG images: {white_image_count}")
