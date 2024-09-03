import pandas as pd
import os

def clean_data(file_path):
    try:
        # Load the dataset
        data = pd.read_csv(file_path)

        # Remove rows where 'Measurement' column is empty
        data_cleaned = data.dropna(subset=['Measurement'])

        # Save the cleaned data back to the same file
        data_cleaned.to_csv(file_path, index=False)
        print(f"Cleaned data in file: {file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "sorted_processed" in file:
                file_path = os.path.join(root, file)
                clean_data(file_path)

# Specify your directory
directory = 'bankAppData/all_data/entryActivity'  # Replace with your directory path
process_directory(directory)
