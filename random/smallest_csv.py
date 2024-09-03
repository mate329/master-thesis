import os
import pandas as pd

def delete_empty_csvs(directory, sensor_types):
    # Walk through all directories and files in the specified directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "entryActivity" in file and file.endswith(".csv"):
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file using pandas
                    df = pd.read_csv(file_path)
                    # Check if the 'sensorType' column exists and if so, check for empty entries for the specified types
                    if 'sensorType' in df.columns and df[df['sensorType'].isin(sensor_types)].empty:
                        # If CSV has 0 entries for the specified sensor types, delete the file
                        os.remove(file_path)
                        print(f"Deleted {file_path} as it had no entries for specified sensor types.")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# List of gyroscope types to check for empty entries
gyroscope_types = [
    'st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE',
    'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro'
]

# Example usage: replace '/path/to/directory' with the path to the directory you want to clean
delete_empty_csvs('bankAppData_original', gyroscope_types)
