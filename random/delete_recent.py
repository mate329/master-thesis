import os
import time

def delete_recent_files(directory, age_minutes=180):
    # Get the current time
    current_time = time.time()
    
    # Convert age in minutes to seconds
    age_seconds = age_minutes * 60
    
    # Walk through all directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Get the path to the file
            file_path = os.path.join(root, file)
            
            # Get the creation time of the file
            creation_time = os.path.getctime(file_path)
            
            # Check if the file was created within the last 'age_minutes' minutes
            if (current_time - creation_time) < age_seconds:
                # If yes, delete the file
                os.remove(file_path)
                print(f"Deleted {file_path}")

# Example usage: replace '/path/to/directory' with the path to the directory you want to clean
delete_recent_files('spectograms_entryActivity', age_minutes=180)
