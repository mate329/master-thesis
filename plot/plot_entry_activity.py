import pandas as pd
import plotly.express as px
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

def valid_date_range(date_series):
    """ Check if date range is valid to avoid large integer conversion """
    try:
        min_date = pd.to_datetime('1900-01-01')
        max_date = pd.to_datetime('2100-01-01')
        return date_series.between(min_date, max_date).all()
    except:
        return False

def process_and_plot(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['date'])

        if not is_string_dtype(df['sensorType']):
            print(f"Skipping file (sensorType not a string): {file_path}")
            return

        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y/%H:%M:%S.%f')
        if not valid_date_range(df['date']):
            print(f"Invalid date range in file: {file_path}")
            return

        df['mappedSensorType'] = df['sensorType'].map(map_sensor_type)

        for sensor_type in ['gyroscope', 'accelerometer']:
            sensor_data = df[df['mappedSensorType'] == sensor_type]
            if not sensor_data.empty:
                fig = px.line(sensor_data, x='date', y=['angularSpeedX', 'angularSpeedY', 'angularSpeedZ'],
                              title=f'{sensor_type.capitalize()} Data over Time')
                save_path = f"{os.path.splitext(file_path)[0]}_{sensor_type}.png"
                fig.write_image(save_path)
                print(f"Plot saved: {save_path}")
            else:
                print(f"No data for '{sensor_type}' in file: {file_path}")

    except OverflowError:
        print(f"Overflow error in file: {file_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def search_and_process(directory):
    # Get all file paths first
    file_paths = glob.glob(f'{directory}/**/*entryActivity*.csv', recursive=True)

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        process_and_plot(file_path)

# Start the process
search_directory = '../bankAppData/'  # Replace with your directory path
search_and_process(search_directory)
