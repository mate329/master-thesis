import pandas as pd
import os

def process_angular_speed_data(file_path, output_file_path):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Check for required columns
        required_columns = ['username', 'angularSpeedX', 'angularSpeedY', 'angularSpeedZ']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the file {file_path}: {', '.join(missing_columns)}")

        # Identify unique users and create a mapping to user numbers
        unique_users = df['username'].unique()
        user_mapping = {user: idx + 1 for idx, user in enumerate(unique_users)}

        # Prepare a list of new dataframes to be concatenated
        new_dfs = []

        for user in unique_users:
            user_df = df[df['username'] == user].copy()
            user_day = user_mapping[user]
            for axis in ['X', 'Y', 'Z']:
                user_df[f'angularSpeed{axis}_day{user_day}'] = user_df[f'angularSpeed{axis}']
            new_dfs.append(user_df)

        # Concatenate all new dataframes
        final_df = pd.concat(new_dfs, axis=0)

        # Drop all columns except the new angular speed columns and then the first three columns
        cols_to_keep = [col for col in final_df.columns if col.startswith('angularSpeed')]
        final_df = final_df[cols_to_keep]
        final_df = final_df.iloc[:, 3:]

        # Save the modified data to a new CSV file
        final_df.to_csv(output_file_path, index=False)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if "gyroscope_entryActivity" in file and "processed" not in file and "sorted" not in file:
                print(f"\nProcessing {file}")
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(root, 'processed_' + file)
                process_angular_speed_data(input_file_path, output_file_path)
                print(f"Processed {file}")

# Specify your directory
directory = 'bankAppData/all_data/entryActivity'  # Replace with your directory path
process_directory(directory)
