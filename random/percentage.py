import os

def is_significant_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Extract the P-value from the ANOVA summary (assuming it's the 5th number in the file)
            p_value_line = lines[4]
            p_value = float(p_value_line.split()[4])
            print(f"File {file_path} has P-value: {p_value}")
            return p_value < 0.05
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return False

def process_directory_for_significance(directory):
    gyro_significant, accel_significant, gyro_total, accel_total = 0, 0, 0, 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "ANOVA_RM_RESULTS" in file and file.endswith(".txt") and "10x" not in file:
                sensor_type = 'gyroscope' if 'gyroscope' in file.lower() else 'accelerometer'
                file_path = os.path.join(root, file)

                if is_significant_from_file(file_path):
                    if sensor_type == 'gyroscope':
                        gyro_significant += 1
                    else:
                        print(f"File {file_path} is significant")
                        accel_significant += 1
                
                if sensor_type == 'gyroscope':
                    gyro_total += 1
                else:
                    accel_total += 1

    # Calculate and print percentage of statistical relevance
    gyro_percent = (gyro_significant / gyro_total * 100) if gyro_total > 0 else 0
    accel_percent = (accel_significant / accel_total * 100) if accel_total > 0 else 0
    print(f"Gyroscope Not Statistically Significant: {100 - gyro_percent:.2f}%")
    print(f"Accelerometer Not Statistically Significant: {100 - accel_percent:.2f}%")

# Specify your directory
directory = 'bankAppData/all_data/entryActivity'  # Replace with your directory path
process_directory_for_significance(directory)
