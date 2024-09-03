import os
import numpy as np
import pandas as pd
from collections import defaultdict

def process_sensor_data_for_user(user_dirs, gyroscope_types, accelerometer_types):
    results = {'accelerometer': {'avg': [], 'std': []}, 'gyroscope': {'avg': [], 'std': []}}

    for directory in user_dirs:
        print(f"Processing directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if 'enterPINactivity' in file and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    df = pd.read_csv(file_path, index_col=False)

                    # Process each sensor type
                    for sensor_label, sensor_types in [('accelerometer', accelerometer_types), ('gyroscope', gyroscope_types)]:
                        for sensor_type in sensor_types:
                            sensor_df = df[df['sensorType'] == sensor_type]
                            if sensor_df.empty:
                                continue
                            
                            sensor_avg = sensor_df[['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']].mean().values
                            sensor_std = sensor_df[['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']].std().values
                            
                            results[sensor_label]['avg'].append(sensor_avg)
                            results[sensor_label]['std'].append(sensor_std)

    # Calculate overall averages and standard deviations
    final_results = []
    for sensor in ['accelerometer', 'gyroscope']:
        for stat in ['avg', 'std']:
            if results[sensor][stat]:
                overall_stat = np.mean(np.vstack(results[sensor][stat]), axis=0)
                final_results.extend(overall_stat)
            else:
                final_results.extend([np.nan, np.nan, np.nan]) # Append NaN if no data is available

    return final_results

# The parent directory containing all recordings
parent_directory = "../../bankAppData"

gyroscope_types = ['bmi26x', 'st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE', 'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro']
accelerometer_types = ['bmi26x', 'Acceleration', 'accelerometer-lsm6ds3-c', 'icm40607_acc', 'accelerometer-lsm6dsm', 'accelerometer-bmi160', 'ACCELEROMETER', 'Accelerometer', 'accelerometer-lsm6ds3', 'LSM6DSO']


# Identify unique users and their directories
user_dirs = defaultdict(list)
for root, dirs, files in os.walk(parent_directory):
    for dir in dirs:
        username = '_'.join(dir.split('_')[:2])
        user_dirs[username].append(os.path.join(root, dir))

# Process data for each user and compile into a single DataFrame
all_user_results = []
for user, dirs in user_dirs.items():
    print(f"Processing data for user: {user}")
    user_results = process_sensor_data_for_user(dirs, gyroscope_types, accelerometer_types)
    all_user_results.append([user] + user_results)

# Create a DataFrame from the results
final_df = pd.DataFrame(all_user_results, columns=['username', 
                                                   'entryActivity_accelerometer_avg_x', 'entryActivity_accelerometer_avg_y', 'entryActivity_accelerometer_avg_z',
                                                   'entryActivity_accelerometer_std_x', 'entryActivity_accelerometer_std_y', 'entryActivity_accelerometer_std_z',
                                                   'entryActivity_gyroscope_avg_x', 'entryActivity_gyroscope_avg_y', 'entryActivity_gyroscope_avg_z',
                                                   'entryActivity_gyroscope_std_x', 'entryActivity_gyroscope_std_y', 'entryActivity_gyroscope_std_z'])

# Saving the final summary DataFrame to a CSV file
output_file = os.path.join(parent_directory, 'entryActivity_results.csv')
final_df.to_csv(output_file, index=False)

print(f"Overall summary results saved to {output_file}")
