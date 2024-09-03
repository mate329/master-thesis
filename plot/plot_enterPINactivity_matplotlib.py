import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import time

def is_string_dtype(series):
    """ Check if a pandas Series has string data type """
    return pd.api.types.is_string_dtype(series)

def map_sensor_type(sensor_type_str):
    gyroscope_types = ['st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE', 'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro']
    accelerometer_types = ['Acceleration', 'accelerometer-lsm6ds3-c', 'icm40607_acc', 'accelerometer-lsm6dsm', 'accelerometer-bmi160', 'ACCELEROMETER', 'Accelerometer', 'accelerometer-lsm6ds3']

    if sensor_type_str in gyroscope_types:
        return 'gyroscope'
    elif sensor_type_str in accelerometer_types:
        return 'accelerometer'
    return None

def process_and_plot(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])

        if not is_string_dtype(df['sensorType']):
            print(f"Skipping file (sensorType not a string): {file_path}")
            return

        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y/%H:%M:%S.%f')
        df['mappedSensorType'] = df['sensorType'].map(map_sensor_type)

        for sensor_type in ['gyroscope', 'accelerometer']:
            sensor_data = df[df['mappedSensorType'] == sensor_type]
            if not sensor_data.empty:
                plt.figure()
                plt.plot(sensor_data['date'], sensor_data['angularSpeedX'], label='X')
                plt.plot(sensor_data['date'], sensor_data['angularSpeedY'], label='Y')
                plt.plot(sensor_data['date'], sensor_data['angularSpeedZ'], label='Z')
                plt.xlabel('Date')
                plt.ylabel('Angular Speed')
                plt.title(f'{sensor_type.capitalize()} Data over Time')
                plt.legend()
                save_path = f"{os.path.splitext(file_path)[0]}_{sensor_type}.png"
                plt.savefig(save_path)
                plt.close()
                print(f"Plot saved: {save_path}")
            else:
                print(f"No data for '{sensor_type}' in file: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def search_and_process(directory):
    # Get all file paths first
    file_paths = glob.glob(f'{directory}/**/*enterPINactivity*.csv', recursive=True)

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        process_and_plot(file_path)

        # Add a delay of 5 seconds between each file processing (optional)
        time.sleep(5)

# Start the process
search_directory = '../bankAppData/'  # Replace with your directory path
search_and_process(search_directory)
