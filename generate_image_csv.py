import os
import csv

# Define the paths for the train, valid, and test image directories
train_dir = "./spectograms/spectrograms_enterPINactivity/train"
valid_dir = "./spectograms/spectrograms_enterPINactivity/valid"
test_dir = "./spectograms/spectrograms_enterPINactivity/test"

# CSV file path for image data
csv_file_path = "./enterpinactivity_image_csv.csv"
# CSV file path for user ID and full names mapping
user_mapping_file_path = "./enterpinactivity_user_id_mapping.csv"

# Headers for the image CSV file
headers = ['image_path', 'dataset_type', 'label_type', 'user_id']

# Initialize a dictionary to keep track of user names and their corresponding IDs
user_id_mapping = {}
current_user_id = 1  # Start user ID numbering from 1

# Function to get or create a numeric user ID for a given name
def get_user_id(name):
    global current_user_id
    if name not in user_id_mapping:
        user_id_mapping[name] = current_user_id
        current_user_id += 1
    return user_id_mapping[name]

# Open the CSV file in write mode to write image data
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the header

    # Process each directory (train, valid, and test)
    for dataset_type, directory in [('train', train_dir), ('valid', valid_dir), ('test', test_dir)]:
        full_directory_path = os.path.abspath(directory)  # Get the full path to the directory
        if os.path.exists(full_directory_path):  # Check if the directory exists
            for img in os.listdir(full_directory_path):
                if img.endswith(('.png', '.jpg', '.jpeg')):  # Filter for image files
                    # Extract label type from the filename
                    label_type = ''
                    if 'angularSpeedX' in img:
                        label_type = 'X'
                    elif 'angularSpeedY' in img:
                        label_type = 'Y'
                    elif 'angularSpeedZ' in img:
                        label_type = 'Z'
                    
                    # Extract user name from the filename
                    name_part = img.split('-')[1].split('_')  # Split at hyphen, then underscore
                    user_name = ' '.join(name_part[:2])  # First two elements are the name and surname

                    # Get or create a numeric user ID
                    user_id = get_user_id(user_name)

                    # Construct the full image path
                    full_image_path = os.path.join(full_directory_path, img)

                    # Write to the CSV file
                    writer.writerow([full_image_path, dataset_type, label_type, user_id])

# Save the user ID and full names mapping to a separate CSV file
with open(user_mapping_file_path, 'w', newline='') as mapping_file:
    mapping_writer = csv.writer(mapping_file)
    mapping_writer.writerow(['user_id', 'full_name'])  # Header for mapping CSV
    for full_name, user_id in user_id_mapping.items():
        mapping_writer.writerow([user_id, full_name])

print("Data processing complete. Image data saved to", csv_file_path)
print("User ID mapping saved to", user_mapping_file_path)
