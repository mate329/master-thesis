import os
import pandas as pd
import numpy as np

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def process_file(file_path):
    try:
        # Load the file
        data = pd.read_csv(file_path)

        # Calculate distance and time difference
        data['shifted_eventRawX'] = data['eventRawX'].shift(-1)
        data['shifted_eventRawY'] = data['eventRawY'].shift(-1)
        data['shifted_eventTime'] = data['eventTime'].shift(-1)
        data['distance'] = calculate_distance(data['eventRawX'], data['eventRawY'], 
                                              data['shifted_eventRawX'], data['shifted_eventRawY'])
        data['time_diff'] = data['shifted_eventTime'] - data['eventTime']
        data = data[:-1]  # Remove last row
        data['speed_pixels_per_ms'] = data['distance'] / data['time_diff']

        # Prepare output data
        output_data = data[['eventRawX', 'eventRawY', 'eventTime', 'distance', 'time_diff', 'speed_pixels_per_ms']]

        # Define output file path
        output_file_path = os.path.splitext(file_path)[0] + '_speed_output.csv'

        # Write to Excel
        output_data.to_csv(output_file_path, index=False)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def search_and_process_files(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if "enterPINxyClick" in file and file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                process_file(file_path)
                print(f"Processed: {file_path}")

search_and_process_files('../bankAppData')
