import os
import shutil
from sklearn.model_selection import train_test_split

def distribute_files(root_dir):
    # Dictionaries to hold file paths for each user and axis type
    user_files = {}

    # Recursively walk through the directory structure
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if "angularSpeed" in file:
                user = file.split('_')[0]  # Assuming the user identifier is the first part of the filename
                if user not in user_files:
                    user_files[user] = {'X': [], 'Y': [], 'Z': []}
                if "angularSpeedX" in file:
                    user_files[user]['X'].append(os.path.join(subdir, file))
                elif "angularSpeedY" in file:
                    user_files[user]['Y'].append(os.path.join(subdir, file))
                elif "angularSpeedZ" in file:
                    user_files[user]['Z'].append(os.path.join(subdir, file))

    # Create train, valid, and test directories if they do not exist
    for data_split in ['train', 'valid', 'test']:
        split_dir = os.path.join(root_dir, data_split)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

    # Helper function to copy and store files
    def copy_and_store_files(file_list, train_dir, valid_dir, test_dir):
        if len(file_list) > 0:
            # Ensure each user has an 80:10:10 split
            try:
                train_files, temp_files = train_test_split(file_list, test_size=0.20, random_state=42)
                valid_files, test_files = train_test_split(temp_files, test_size=0.50, random_state=42)
            except ValueError:
                print(f"Skipping user {user} due to insufficient data.")
                return

            for file in train_files:
                shutil.copy(file, os.path.join(train_dir, os.path.basename(file)))
            for file in valid_files:
                shutil.copy(file, os.path.join(valid_dir, os.path.basename(file)))
            for file in test_files:
                shutil.copy(file, os.path.join(test_dir, os.path.basename(file)))

    # Common directories for train, valid, and test
    train_dir = os.path.join(root_dir, 'train')
    valid_dir = os.path.join(root_dir, 'valid')
    test_dir = os.path.join(root_dir, 'test')

    # Iterate over each user and axis type to distribute files
    for user, axes_files in user_files.items():
        for axis, files in axes_files.items():
            # Distribute files for each axis type
            copy_and_store_files(files, train_dir, valid_dir, test_dir)

# Example usage:
# Replace './spectograms/spectrograms_enterPINactivity' with the path to your data folder.
distribute_files('./spectograms/spectrograms_enterPINactivity')
