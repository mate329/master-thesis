import os
import pandas as pd

def process_csv_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'e2RawY' in df.columns:
            # Remove everything after the dot in 'e2RawY' column
            df['e2RawY'] = df['e2RawY'].astype(str).str.split('.').str[0]
            df.to_csv(file_path, index=False)
            print(f"Processed {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def find_and_process_files(start_path):
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if 'onScroll' in file and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                process_csv_file(file_path)

# Starting directory - replace with the path where you want to start searching
start_directory = '../bankAppData/'
find_and_process_files(start_directory)
