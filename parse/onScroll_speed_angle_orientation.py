import os
import pandas as pd
import numpy as np
import math

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate the angle of the swipe
def calculate_swipe_angle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    angle_radians = math.atan2(dy, dx)
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees

# Function to determine the orientation of the swipe
def determine_swipe_orientation(x1, y1, x2, y2, threshold=50):
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) > threshold and abs(dy) <= threshold:
        return "Left to Right" if dx > 0 else "Right to Left"
    elif abs(dy) > threshold and abs(dx) <= threshold:
        return "Top to Bottom" if dy > 0 else "Bottom to Top"
    else:
        return "Diagonal"

def process_swipe_file_with_angle_and_orientation(file_path):
    try:
        # Load the file
        data = pd.read_csv(file_path)

        # Calculate distance, time difference, angle, and orientation for swipe
        data['distance'] = calculate_distance(data['e1RawX'], data['e1RawY'], data['e2RawX'], data['e2RawY'])
        data['time_diff'] = data['e2EventTime'] - data['e1EventTime']
        data['speed_pixels_per_ms'] = data['distance'] / data['time_diff']
        data['swipe_angle'] = data.apply(lambda row: calculate_swipe_angle(row['e1RawX'], row['e1RawY'], row['e2RawX'], row['e2RawY']), axis=1)
        data['swipe_orientation'] = data.apply(lambda row: determine_swipe_orientation(row['e1RawX'], row['e1RawY'], row['e2RawX'], row['e2RawY']), axis=1)

        # Prepare output data
        output_data = data[['e1RawX', 'e1RawY', 'e2RawX', 'e2RawY', 'e1EventTime', 'e2EventTime', 
                            'distance', 'time_diff', 'speed_pixels_per_ms', 'swipe_angle', 'swipe_orientation']]

        # Define output file path
        output_file_path = os.path.splitext(file_path)[0] + '_swipe_speed_angle_orientation_output.xlsx'

        # Write to Excel
        output_data.to_excel(output_file_path, index=False)
    except Exception as e:
        print(f"Error processing swipe file {file_path}: {e}")

def search_and_process_swipe_files_with_angle_and_orientation(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if "onScroll" in file and file.lower().endswith('.csv'):
                file_path = os.path.join(subdir, file)
                process_swipe_file_with_angle_and_orientation(file_path)
                print(f"Processed swipe file with angle and orientation: {file_path}")

search_and_process_swipe_files_with_angle_and_orientation('../bankAppData')
