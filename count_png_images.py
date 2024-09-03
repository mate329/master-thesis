import os

def count_png_images(directory):
    """
    Recursively finds and counts all PNG files in the specified directory.

    Parameters:
    directory (str): The path of the directory to search for PNG files.

    Returns:
    int: The total number of PNG images found.
    """
    png_image_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_image_count += 1
                print(f"Found PNG image: {os.path.join(root, file)}")

    return png_image_count

# Example usage
directory = './spectograms/spectrograms_enterPINactivity'  # Replace with the path to your directory
total_png_images = count_png_images(directory)
print(f"Total number of PNG images: {total_png_images}")
