import os
from PIL import Image
import numpy as np

def delete_white_png_images(directory):
    """
    Recursively finds PNG files in the directory and deletes those that are completely white.

    Parameters:
    directory (str): The path of the directory to search for PNG files.
    """
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                
                try:
                    # Open the image file
                    with Image.open(file_path) as img:
                        # Convert image to numpy array
                        img_array = np.array(img)
                        
                        # Check if the image is completely white
                        if np.all(img_array == 255):
                            # If so, delete the image file
                            os.remove(file_path)
                            print(f"Deleted completely white image: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

# Example usage
directory = './spectograms/spectrograms_enterPINactivity'  # Replace with the path to your directory

delete_white_png_images(directory)
