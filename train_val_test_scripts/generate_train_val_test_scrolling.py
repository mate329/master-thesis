import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_and_save(train_data, valid_data, test_data, sensor_type, directory):
    # Save the datasets into separate files
    train_data.to_csv(os.path.join(directory, f"train_{sensor_type}.csv"), index=False)
    valid_data.to_csv(os.path.join(directory, f"valid_{sensor_type}.csv"), index=False)
    test_data.to_csv(os.path.join(directory, f"test_{sensor_type}.csv"), index=False)

def process_files(input_directory, output_directory):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Initialize empty dataframes for each sensor type
    train_scrolling, valid_scrolling, test_scrolling = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if 'scrollingActivity' in file and file.endswith('.csv'):
                # Full path to the file
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)
                print(f'Processing file {file_path}')

                s_train, temp_scroll = train_test_split(data, test_size=0.4, random_state=42)
                s_valid, s_test = train_test_split(temp_scroll, test_size=0.5, random_state=42)
                train_scrolling = pd.concat([train_scrolling, s_train])
                valid_scrolling = pd.concat([valid_scrolling, s_valid])
                test_scrolling = pd.concat([test_scrolling, s_test])

                print(f"Processed {file_path}")
            
        # Save the aggregated datasets
        if not train_scrolling.empty:
            split_and_save(train_scrolling, valid_scrolling, test_scrolling, 'scrolling', output_directory)

root_directory = '../bankAppData/'
output_directory = os.path.join(root_directory, '0scrollingActivityDatasets')
process_files(root_directory, output_directory)