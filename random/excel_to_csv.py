import os
import pandas as pd

def convert_excel_to_csv(directory):
    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check for Excel files with .xlsx extension
            if "swipe_speed_angle_orientation_output" in file and file.endswith('.xlsx'):
                excel_file_path = os.path.join(root, file)
                # Define the CSV file path (same name as the Excel file, with .csv extension)
                csv_file_path = os.path.splitext(excel_file_path)[0] + '.csv'
                
                try:
                    # Read the Excel file, specify engine if needed
                    df = pd.read_excel(excel_file_path, engine='openpyxl')
                    
                    # Save the dataframe to a CSV file
                    df.to_csv(csv_file_path, index=False)
                    
                    print(f"Converted {excel_file_path} to {csv_file_path}")
                except Exception as e:
                    print(f"Failed to convert {excel_file_path} due to: {e}")

# The parent directory containing all Excel files
parent_directory = 'bankAppData'

# Call the function with the parent directory
convert_excel_to_csv(parent_directory)
