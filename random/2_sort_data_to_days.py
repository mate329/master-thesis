import pandas as pd
import os

def sort_data(file_path, output_file_path):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Melt the data to long format
        melted_data = pd.melt(data, var_name='Day_Axis', value_name='Measurement')

        # Split 'Day_Axis' into separate 'Day' and 'Axis' columns
        melted_data[['Axis', 'Day']] = melted_data['Day_Axis'].str.split('_', expand=True)
        melted_data.drop(columns=['Day_Axis'], inplace=True)

        # Reorder columns
        melted_data = melted_data[['Day', 'Axis', 'Measurement']]

        # Save the transformed data
        melted_data.to_csv(output_file_path, index=False)

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "processed" in file:
                print(f"\n Processing {file}")
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(root, 'sorted_' + file)
                sort_data(input_file_path, output_file_path)
                print(f"Processed and sorted {file}")

# Specify your directory
directory = 'bankAppData/all_data/entryActivity'  # Replace with your directory path
process_directory(directory)
