import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_boxplot_stats(sensor_data, sensor_name, file_path):
    # Generating box plot
    plt.figure(figsize=(8, 6))
    plt.boxplot([sensor_data[col] for col in sensor_data.columns], labels=sensor_data.columns)
    plt.title(f'Box Plot of {sensor_name} Values')
    plt.ylabel('Values')
    plt.grid(True)
    plt.tight_layout()

    # Save the box plot to a file
    boxplot_file_path = os.path.splitext(file_path)[0] + f'_{sensor_name.lower()}_boxplot.png'
    plt.savefig(boxplot_file_path)
    plt.close()
    return boxplot_file_path

def process_activity_file_with_boxplot(file_path):
    data = pd.read_csv(file_path, index_col=False)
    gyro_data = data[data['sensorType'] == 'Gyroscope'][['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']]
    accel_data = data[data['sensorType'] == 'Acceleration'][['angularSpeedX', 'angularSpeedY', 'angularSpeedZ']]

    gyro_boxplot_path = plot_boxplot_stats(gyro_data, 'Gyroscope', file_path)
    accel_boxplot_path = plot_boxplot_stats(accel_data, 'Acceleration', file_path)
    
    return gyro_boxplot_path, accel_boxplot_path

def search_and_process_activity_files_with_boxplot(root_dir):
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if "enterPINactivity" in file and file.lower().endswith('.csv'):
                file_path = os.path.join(subdir, file)
                gyro_boxplot_path, accel_boxplot_path = process_activity_file_with_boxplot(file_path)
                print(f"Processed activity file and generated gyro box plot: {gyro_boxplot_path}")
                print(f"Processed activity file and generated accel box plot: {accel_boxplot_path}")



search_and_process_activity_files_with_boxplot('../bankAppData')
