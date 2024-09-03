import os
import pandas as pd
from collections import defaultdict

gyroscope_types = ['bmi26x', 'st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE', 'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro']
accelerometer_types = ['bmi26x', 'Acceleration', 'accelerometer-lsm6ds3-c', 'icm40607_acc', 'accelerometer-lsm6dsm', 'accelerometer-bmi160', 'ACCELEROMETER', 'Accelerometer', 'accelerometer-lsm6ds3', 'LSM6DSO']

def append_csv_files_by_user_and_sensor(parent_directory, output_directory):
    user_gyro_data = defaultdict(list)
    user_accel_data = defaultdict(list)

    for root, _, files in os.walk(parent_directory):
        for file in files:
            if 'entryActivity' in file and file.endswith('.csv'): # enterPinActivity, entryActivity
                file_path = os.path.join(root, file)
                folder_name = os.path.basename(root)
                username = '_'.join(folder_name.split('_')[:2])
                df = pd.read_csv(file_path, index_col=False)

                if 'sensorType' in df.columns:
                    # Split data into gyroscope and accelerometer dataframes
                    gyro_df = df[df['sensorType'].isin(gyroscope_types)]
                    accel_df = df[df['sensorType'].isin(accelerometer_types)]

                    if not gyro_df.empty:
                        user_gyro_data[username].append(gyro_df)
                    if not accel_df.empty:
                        user_accel_data[username].append(accel_df)
                else:
                    print(f"Warning: 'sensorType' column not found in {file}")

    # Process and save data for each user and sensor type
    for username in set(list(user_gyro_data.keys()) + list(user_accel_data.keys())):
        user_output_directory = os.path.join(output_directory, username)
        if not os.path.exists(user_output_directory):
            os.makedirs(user_output_directory)

        # Gyroscope data
        if user_gyro_data[username]:
            save_sensor_data(user_gyro_data, username, user_output_directory, 'gyroscope')
        
        # Accelerometer data
        if user_accel_data[username]:
            save_sensor_data(user_accel_data, username, user_output_directory, 'accelerometer')

def save_sensor_data(user_sensor_data, username, user_output_directory, sensor_type):
    concatenated_df = pd.concat(user_sensor_data[username], ignore_index=True)
    output_file = os.path.join(user_output_directory, f'{username}_{sensor_type}_entryActivity.csv')
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        concatenated_df = pd.concat([existing_df, concatenated_df], ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Data for {username} ({sensor_type}) saved or updated in {output_file}")

# Directories
parent_directory = "bankAppData"
output_directory = "bankAppData/all_data/entryActivity"

# Run the function
append_csv_files_by_user_and_sensor(parent_directory, output_directory)
