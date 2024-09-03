import pandas as pd
import glob

def is_string_dtype(series):
    """ Check if a pandas Series has string data type """
    return pd.api.types.is_string_dtype(series)

def find_unique_sensor_types(file_path, unique_sensor_types_set):
    try:
        df = pd.read_csv(file_path)

        # Check if sensorType is of string data type
        if not is_string_dtype(df['sensorType']):
            print(f"Skipping file (sensorType not a string): {file_path}")
            return

        unique_sensor_types_in_file = set(df['sensorType'].unique())
        unique_sensor_types_set.update(unique_sensor_types_in_file)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def search_files_and_find_sensor_types(directory):
    unique_sensor_types_set = set()

    for file_path in glob.glob(f'{directory}/**/*enterPINactivity*.csv', recursive=True):
        find_unique_sensor_types(file_path, unique_sensor_types_set)

    print(f"Overall Unique Sensor Types in All Files: {unique_sensor_types_set}")

# Start the process
search_directory = 'bankAppData'  # Replace with your directory path
search_files_and_find_sensor_types(search_directory)
