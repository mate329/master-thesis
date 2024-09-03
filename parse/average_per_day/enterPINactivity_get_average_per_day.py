import os
import numpy as np
import pandas as pd
from collections import defaultdict

def process_sensor_data_for_day(directories, gyroscope_types, accelerometer_types):
    daily_results = {'accelerometer': {'avg': [], 'std': []}, 'gyroscope': {'avg': [], 'std': []}}

    for directory in directories:
        print(f"Processing directory: {directory}")
        for root, dirs, files in os.walk(directory):
            for file in files:
                if 'enterPINactivity' in file and file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")  # Additional print statement
                    df = pd.read_csv(file_path, index_col=False)

                    # Process each sensor type
                    for sensor_label, sensor_types in [('accelerometer', accelerometer_types), ('gyroscope', gyroscope_types)]:
                        for sensor_type in sensor_types:
                            sensor_df = df[df['sensorType'] == sensor_type]
                            if sensor_df.empty:
                                continue
                            
                            sensor_avg = sensor_df[['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']].mean().values
                            sensor_std = sensor_df[['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']].std().values
                            
                            daily_results[sensor_label]['avg'].append(sensor_avg)
                            daily_results[sensor_label]['std'].append(sensor_std)

    # Calculate daily averages and standard deviations
    final_daily_results = []
    for sensor in ['accelerometer', 'gyroscope']:
        for stat in ['avg', 'std']:
            if daily_results[sensor][stat]:
                overall_stat = np.mean(np.vstack(daily_results[sensor][stat]), axis=0)
                final_daily_results.extend(overall_stat)
            else:
                final_daily_results.extend([np.nan, np.nan, np.nan])  # Append NaN if no data is available

    return final_daily_results


# The parent directory containing all recordings
parent_directory = "../../bankAppData"

gyroscope_types = ['bmi26x', 'st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE', 'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro']
accelerometer_types = ['bmi26x', 'Acceleration', 'accelerometer-lsm6ds3-c', 'icm40607_acc', 'accelerometer-lsm6dsm', 'accelerometer-bmi160', 'ACCELEROMETER', 'Accelerometer', 'accelerometer-lsm6ds3', 'LSM6DSO']

# Identify unique days for each user
user_daily_dirs = defaultdict(lambda: defaultdict(list))
for root, dirs, files in os.walk(parent_directory):
    for dir in dirs:
        parts = dir.split('_')
        username = '_'.join(parts[:2])
        timestamp = parts[-1]
        user_daily_dirs[username][timestamp].append(os.path.join(root, dir))

# Process data for each day for each user and compile into a single DataFrame
all_user_daily_results = []
for user, daily_dirs in user_daily_dirs.items():
    for day, dirs in daily_dirs.items():
        print(f"Processing data for user: {user} on day: {day}")
        day_results = process_sensor_data_for_day(dirs, gyroscope_types, accelerometer_types)
        all_user_daily_results.append([user, day] + day_results)

# Create a DataFrame from the results
columns = ['username', 'day',
           'enterPINactivity_accelerometer_avg_x', 'enterPINactivity_accelerometer_avg_y', 'enterPINactivity_accelerometer_avg_z',
           'enterPINactivity_accelerometer_std_x', 'enterPINactivity_accelerometer_std_y', 'enterPINactivity_accelerometer_std_z',
           'enterPINactivity_gyroscope_avg_x', 'enterPINactivity_gyroscope_avg_y', 'enterPINactivity_gyroscope_avg_z',
           'enterPINactivity_gyroscope_std_x', 'enterPINactivity_gyroscope_std_y', 'enterPINactivity_gyroscope_std_z']
final_df = pd.DataFrame(all_user_daily_results, columns=columns)

# Saving the final summary DataFrame to a CSV file
output_file = os.path.join(parent_directory, 'daily_enterPINactivity_results.csv')
final_df.to_csv(output_file, index=False)

print(f"Daily summary results saved to {output_file}")
