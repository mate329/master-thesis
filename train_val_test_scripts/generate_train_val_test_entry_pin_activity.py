import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Lists of possible gyroscope and accelerometer types
gyroscopes = ['bmi26x', 'st-lsm6ds3-c', 'gyroscope-lsm6dsm', 'gyroscope-lsm6ds3', 'GYROSCOPE', 'Gyroscope', 'gyroscope-lsm6ds3-c', 'icm40607_gyro']
accelerometers = ['bmi26x', 'Acceleration', 'accelerometer-lsm6ds3-c', 'icm40607_acc', 'accelerometer-lsm6dsm', 'accelerometer-bmi160', 'ACCELEROMETER', 'Accelerometer', 'accelerometer-lsm6ds3', 'LSM6DSO']

def split_and_save(train_data, valid_data, test_data, sensor_type, directory):
    # Save the datasets into separate files
    train_data.to_csv(os.path.join(directory, f"train_{sensor_type}.csv"), index=False)
    valid_data.to_csv(os.path.join(directory, f"valid_{sensor_type}.csv"), index=False)
    test_data.to_csv(os.path.join(directory, f"test_{sensor_type}.csv"), index=False)

def process_files(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    filename_pattern = 'entryActivity' if output_directory == 'bankAppData/0entryActivityDatasets' else 'enterPINactivity'

    # Initialize empty dataframes for each sensor type
    train_gyro, valid_gyro, test_gyro = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    train_accel, valid_accel, test_accel = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if filename_pattern in file and file.endswith('.csv'):
                # Full path to the file
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                print(f'Processing file {file_path}')

                # Process and aggregate gyroscope data
                gyro_data = data[data['sensorType'].isin(gyroscopes)]
                if not gyro_data.empty:
                    g_train, temp_gyro = train_test_split(gyro_data, test_size=0.4, random_state=42)
                    g_valid, g_test = train_test_split(temp_gyro, test_size=0.5, random_state=42)
                    train_gyro = pd.concat([train_gyro, g_train])
                    valid_gyro = pd.concat([valid_gyro, g_valid])
                    test_gyro = pd.concat([test_gyro, g_test])

                # Process and aggregate accelerometer data
                accel_data = data[data['sensorType'].isin(accelerometers)]
                if not accel_data.empty:
                    a_train, temp_accel = train_test_split(accel_data, test_size=0.4, random_state=42)
                    a_valid, a_test = train_test_split(temp_accel, test_size=0.5, random_state=42)
                    train_accel = pd.concat([train_accel, a_train])
                    valid_accel = pd.concat([valid_accel, a_valid])
                    test_accel = pd.concat([test_accel, a_test])

                print(f"Processed {file_path}")
            
        # Save the aggregated datasets
        if not train_gyro.empty:
            split_and_save(train_gyro, valid_gyro, test_gyro, 'gyroscope', output_directory)
        if not train_accel.empty:
            split_and_save(train_accel, valid_accel, test_accel, 'accelerometer', output_directory)

root_directory = '../spectograms/spectograms_enterPINactivity'
for mode in ['0entryActivityDatasets', '0enterPinActivityDatasets']:
    output_directory = os.path.join(root_directory, mode)
    process_files(root_directory, output_directory)