import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import numpy as np
import os

# gyroscope_types = [
#     'st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE',
#     'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro', 'lsm6dso', 'lsm6dso_gyro', 'LSM6DSO'
# ]
gyroscope_types = ['bmi26x', 'st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE', 'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro']

def load_filtered_data(filepath):
    try:
        # Load the CSV file
        df = pd.read_csv(filepath)

        # Verify and inspect column names; adjust 'date' column name as found in your CSV
        if 'date' not in df.columns:
            raise ValueError("Date column not found in the file.")
        
        # Convert 'date' column to datetime format, ensuring the correct date format is used
        df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y/%H:%M:%S.%f", errors='coerce')

        # Filter out rows where date conversion failed
        if df['date'].isna().any():
            print(f"Date conversion failed for some entries in {filepath}.")
        
        # Filter rows where sensorType is a type of Gyroscope
        df_gyro = df[df['sensorType'].isin(gyroscope_types)]
        df_gyro = df[df['sensorType'].isin(gyroscope_types)]

        return df_gyro
    except Exception as e:
        raise Exception(f"Error processing file {filepath}: {str(e)}")

def plot_spectrogram(data, axis_name, filepath):
    try:
        # Calculate time differences in seconds
        time_diffs = data['date'].diff().dt.total_seconds().dropna()

        # Calculate median of time differences and determine the sampling frequency
        fs = 1 / time_diffs.median()

        # Extract the relevant axis data
        axis_data = data[axis_name].values

        # Calculate spectrogram
        f, t, Sxx = spectrogram(axis_data, fs=fs, scaling='spectrum')

        # Plotting the spectrogram
        plt.figure(figsize=(10, 3))  # Set the figure size
        plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        plt.axis('off')  # Turn off axes
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)  # Save the plot without padding and border
        plt.close()
    except Exception as e:
        raise Exception(f"Error plotting data for {axis_name}: {str(e)}")

def process_file(filepath):
    try:
        data = load_filtered_data(filepath)

        # Generate and save a spectrogram for each axis
        for axis in ['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']:
            new_filename = os.path.join(os.path.dirname(filepath), f'spectrogram_{axis}_{os.path.basename(filepath).replace(".csv", ".png")}')
            plot_spectrogram(data, axis, new_filename)

        print(f"Processed and saved files for: {filepath}")
    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

def main():
    root_dir = '../bankAppDataWithPlot'  # Set the path to your directory
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if 'enterPINactivity' in file and file.endswith('.csv'):
                filepath = os.path.join(root, file)
                try:
                    process_file(filepath)
                except Exception as e:
                    print(e)

if __name__ == "__main__":
    main()
