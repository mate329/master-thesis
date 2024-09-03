import os
import pandas as pd

def remove_accelerometer_rows_in_place(directory):
    """
    Recursively finds CSV files in the directory and removes rows where the 'sensorType' is 'Accelerometer'.
    The modified CSV files overwrite the original files.

    Parameters:
    directory (str): The path of the directory to search for CSV files.
    """
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv') and 'enterPINactivity' in file:
                file_path = os.path.join(root, file)
                
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    
                    if 'sensorType' in df.columns:
                        # Drop rows where 'sensorType' is 'Accelerometer'
                        df = df[df['sensorType'] != 'Accelerometer']
                    
                    # Save the modified dataframe to the same file, overwriting it
                    df.to_csv(file_path, index=False)
                    print(f"Processed and overwrote: {file_path}")
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")

# Example usage
directory = './bankAppDataWithPlot'  # Replace with the path to your directory

remove_accelerometer_rows_in_place(directory)
